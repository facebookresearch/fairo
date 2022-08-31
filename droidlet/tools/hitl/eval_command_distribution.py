import argparse
import boto3
import os
import subprocess
import sys
import time
import signal
import glob
import yaml
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from droidlet.tools.hitl.utils.process_s3_logs import read_s3_bucket
from droidlet.dialog.post_process_logical_form import retrieve_ref_obj_span

PROD_BATCH_IDS = [  # The 10 ~1000 HIT jobs from early 2022
    20211115024843,
    20211122224720,
    # 20211202170632, # This one is weird b/c of a bug when launched, see quip
    20211207020233,
    20211230205300,
    20220104011348,
    20220224132033,
    20220228100513,
    20220302172356,
    20220304025542,
]
LABELING_IGLU_BATCHES = [20220804134815, 20220805093118, 20220815162244]
LABELING_SIMPLE_SHAPES_BATCHES = [20220809133448, 20220816110956, 20220816152958]
LABELING_DATA_LUT = {
    "iglu": LABELING_IGLU_BATCHES,
    "simple": LABELING_SIMPLE_SHAPES_BATCHES,
    "both": LABELING_IGLU_BATCHES + LABELING_SIMPLE_SHAPES_BATCHES,
}

ACTION_LIST = ["build", "destroy", "fill", "copy", "dance", "tag", "dig", "spawn", "move"]

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
S3_SYNC_TIMEOUT = 180
S3_POLL_TIME = 5

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)


def scrape_interaction(opts):
    # Determine the batch_ids being scraped
    if opts.test_batch_id:
        batch_ids = [opts.test_batch_id]
    elif opts.just_prod_jobs:
        batch_ids = PROD_BATCH_IDS
    else:
        batch_ids = []
        raise NotImplementedError
    print(f"Producing reference object distribution for batch_ids: {batch_ids}")

    # Pull down the log files to .hitl if not exists and unpack
    for batch_id in batch_ids:
        local_dir = os.path.join(HITL_TMP_DIR, str(batch_id))
        if os.path.isdir(local_dir):
            # FIXME maybe a more robust check if the logs are actually there
            print(f"batch_id {batch_id} data already exists locally")
        else:
            process_s3_logs(batch_id, "interaction")
            print(f"batch_id {batch_id} successfully synced to local folder")

    # Scrape for logical forms, store the reference objects
    ref_objs = []
    for batch_id in batch_ids:
        parsed_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/parsed_turk_logs")
        new_refs = read_turk_logs(parsed_logs_dir)
        if new_refs:
            ref_objs.extend(new_refs)
    print(f"Finished scraping interaction job reference objects: {ref_objs}")

    # Produce some charts
    plot_refs(ref_objs, "interaction")

    data_save_path = os.path.join(opts.output_save_dir, "interaction_refs.pkl")
    with open(data_save_path, "wb") as f:
        pickle.dump(ref_objs, f)

    return ref_objs


def process_s3_logs(batch_id, job_type) -> None:
    """
    This borrows heavily from the function of the same name in
    droidlet.tools.hitl.nsp_retrain.interaction_jobs
    """
    if job_type == "interaction":
        s3_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/turk_logs")
        parsed_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/parsed_turk_logs")
        os.makedirs(parsed_logs_dir, exist_ok=True)
        s3_sync_cmd = f"aws s3 sync {S3_ROOT}/{batch_id}/interaction {s3_logs_dir}"
    else:
        s3_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/vision_labeling")
        s3_sync_cmd = f"aws s3 sync {S3_ROOT}/{batch_id}/vision_labeling_results {s3_logs_dir}"

    os.makedirs(s3_logs_dir, exist_ok=True)

    rc = subprocess.Popen(s3_sync_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)

    now = datetime.now().timestamp()
    while not (datetime.now().timestamp() - now > S3_SYNC_TIMEOUT) and rc.poll() is None:
        print(
            f"S3 syncing for batch_id {batch_id}...Remaining time: {int(S3_SYNC_TIMEOUT - (datetime.now().timestamp() - now))} sec"
        )
        time.sleep(S3_POLL_TIME)

    if rc.poll() is None:
        print("S3 syncing timed out, canceling")
        os.killpg(os.getpgid(rc.pid), signal.SIGINT)
        time.sleep(10)
        os.killpg(os.getpgid(rc.pid), signal.SIGKILL)

    print(f"Done downloading files for batch_id {batch_id}")

    if job_type == "interaction":
        cnt = read_s3_bucket(s3_logs_dir, parsed_logs_dir)  # extract all tarballs
        assert cnt > 0

    return


def read_turk_logs(turk_output_directory) -> list:
    # Crawl turk logs directory and retrieve reference objects
    ref_objs = []
    for csv_path in glob.glob(f"{turk_output_directory}/**/nsp_outputs.csv"):
        csv_file = pd.read_csv(csv_path, delimiter="|")

        for _, row in csv_file.iterrows():
            d = yaml.load(row["action_dict"], yaml.SafeLoader)
            span = d.get("text_span", retrieve_ref_obj_span(d))
            if not span:
                # No ref_obj in this lf, move on
                continue
            new_ref = span_to_text(span, row["command"])
            if new_ref:
                new_ref = new_ref.replace("&nbsp", "")
                new_ref = filter_stop_words(new_ref)
                if new_ref not in ACTION_LIST:
                    # FIXME This issue comes from poor NSP accuracy...
                    ref_objs.append(new_ref)

    return ref_objs


def span_to_text(span, cmd) -> str:
    # Convert span nums to words
    if isinstance(span, list):
        cmd_list = cmd.split(" ")
        return " ".join(cmd_list[span[1][0] : (span[1][1] + 1)])
    elif isinstance(span, str):
        maybe_span_nums = yaml.load(span)
        if isinstance(maybe_span_nums, str):
            return span
        elif isinstance(maybe_span_nums, list):
            span_to_text(maybe_span_nums, cmd)
        else:
            print(f"Invalid span data type! Debug cmd: {cmd} span: {span}")
            return None
    else:
        print(f"Invalid span data type! Debug cmd: {cmd} span: {span}")
        return None


def scrape_labeling(opts):
    # Determine the batch_ids being scraped
    if opts.test_batch_id:
        batch_ids = [opts.test_batch_id]
    else:
        batch_ids = LABELING_DATA_LUT[opts.labeling_job_types]

    # Pull down the log files to .hitl if not exists and unpack
    for batch_id in batch_ids:
        local_dir = os.path.join(HITL_TMP_DIR, str(batch_id))
        if os.path.isdir(local_dir):
            # FIXME maybe a more robust check if the logs are actually there
            print(f"batch_id {batch_id} data already exists locally")
        else:
            process_s3_logs(batch_id, "labeling")
            # FIXME!!!  CSV files not synced to S3, do this by hand and fix for the future
            print(f"batch_id {batch_id} successfully synced to local folder")

    # Scrape for logical forms, store the reference objects
    ref_objs = []
    for batch_id in batch_ids:
        labeling_results_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/vision_labeling")
        new_refs = read_labeling_csv(labeling_results_dir)
        if new_refs:
            ref_objs.extend(new_refs)
    print(f"Finished scraping labeling job reference objects: {ref_objs}")

    # Produce some charts
    plot_refs(ref_objs, f"labeling_{opts.labeling_job_types}")

    data_save_path = os.path.join(opts.output_save_dir, "labeling_refs.pkl")
    with open(data_save_path, "wb") as f:
        pickle.dump(ref_objs, f)

    return ref_objs


def plot_refs(ref_objs, job_type) -> None:
    obj_cnt = Counter(ref_objs)
    most_common = obj_cnt.most_common(50)
    xs = [x[0] for x in most_common]
    ys = [x[1] for x in most_common]
    plt.bar(xs, ys, color="g")
    plt.xlabel("Command")
    plt.xticks(rotation=70, fontsize=6, va="top", ha="right")
    plt.ylabel("Count")
    plt.subplots_adjust(bottom=0.2)
    fig_save_path = os.path.join(opts.output_save_dir, f"{job_type}_ref_obj_cnt.png")
    plt.savefig(fig_save_path, format="png", dpi=300)


def read_labeling_csv(dir):
    # Crawl turk logs directory and retrieve reference objects
    csv_filename = [f for f in os.listdir(dir) if re.match("\d+\.csv", f)][0]
    csv_path = os.path.join(dir, csv_filename)
    csv_file = pd.read_csv(csv_path)

    ref_objs = []
    for _, row in csv_file.iterrows():
        new_ref = filter_stop_words(str(row["object"]))
        if new_ref:
            ref_objs.append(new_ref)

    return ref_objs


def filter_stop_words(ref):
    STRIP_LIST = [
        "a",
        "the",
        "and",
        "an",
        "there",
        "is",
        "here",
        " ",
        "orange",
        "yellow",
        "purple",
        "black",
        "grey",
        "gray",
        "green",
        "white",
        "red",
        "blue",
        "pink",
        "olive",
        "magenta",
        "aqua",
        "lime",
        "cyan",
        "gold",
        "golden",
        "bronze",
        "teal",
        "brown",
        "dark",
        "light",
        "multicolored",
        "hot",
        "colorful",
        "multi-colored",
        "multi",
        "colored",
        "letter",
        "nothing",  # controversial
        "single",
        "big",
        "small",
        "deep",
        "shallow",
        "large",
        "tiny",
        "narrow",
        "wide",
    ]

    def check_filter(word):
        if word not in STRIP_LIST:
            return True
        else:
            return False

    split = ref.split(" ")
    lower = [w.lower() for w in split]
    filtered = filter(check_filter, lower)
    word = " ".join(filtered)
    return word.strip()


def compare_distributions(interaction_refs, labeling_refs):
    # % of each appearing in the other
    interaction_overlaps = 0
    unique_interaction_overlaps = set()
    for ref in interaction_refs:
        if ref in labeling_refs:
            interaction_overlaps += 1
            unique_interaction_overlaps.add(ref)
    print(
        f"% of Interaction ref_objs also in labeling corpus: {interaction_overlaps/len(interaction_refs)}"
    )
    interaction_set = set(interaction_refs)
    print(
        f"% of unique Interaction ref_objs also in labeling corpus: {len(unique_interaction_overlaps)/len(interaction_set)}"
    )

    labeling_overlaps = 0
    unique_labeling_overlaps = set()
    for ref in labeling_refs:
        if ref in interaction_refs:
            labeling_overlaps += 1
            unique_labeling_overlaps.add(ref)
    print(
        f"% of Labeling ref_objs also in interaction corpus: {labeling_overlaps/len(labeling_refs)}"
    )
    labeling_set = set(labeling_refs)
    print(
        f"% of unique Labeling ref_objs also in interaction corpus: {len(unique_labeling_overlaps)/len(labeling_set)}"
    )
    print(unique_labeling_overlaps)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_save_dir", type=str, default="", help="Where to save the output plot"
    )
    parser.add_argument("--test_batch_id", type=int, default=0)
    parser.add_argument(
        "--just_prod_jobs",
        action="store_true",
        default=False,
        help="Scrape just 10 prod jobs from early 2022",
    )
    parser.add_argument(
        "--scrape_interaction",
        action="store_true",
        default=False,
        help="Enable interaction job scrape",
    )
    parser.add_argument(
        "--scrape_labeling", action="store_true", default=False, help="Enable labeling job scrape"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Compare interaction and labeling command distributions",
    )
    parser.add_argument(
        "--labeling_job_types",
        type=str,
        choices=["iglu", "simple", "both"],
        default="simple",
        help="What types of labeling jobs to scrape data from",
    )
    parser.add_argument(
        "--load_existing_data", action="store_true", default=False, help="Load local ref files"
    )
    opts = parser.parse_args()

    interaction_refs, labeling_refs = None, None
    if opts.load_existing_data:
        interaction_path = os.path.join(opts.output_save_dir, "interaction_refs.pkl")
        with open(interaction_path, "rb") as f:
            interaction_refs = pickle.load(f)
        labeling_path = os.path.join(opts.output_save_dir, "labeling_refs.pkl")
        with open(labeling_path, "rb") as f:
            labeling_refs = pickle.load(f)

    if opts.scrape_interaction:
        interaction_refs = scrape_interaction(opts)

    if opts.scrape_labeling:
        labeling_refs = scrape_labeling(opts)

    if interaction_refs and labeling_refs and opts.compare:
        compare_distributions(interaction_refs, labeling_refs)
