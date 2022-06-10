import os
import boto3
import botocore
import tarfile
import re
import pandas as pd
import logging

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)

batch_id = 20220224132033
batch_prefix = f"{batch_id}/interaction/"
bucket = s3.Bucket(S3_BUCKET_NAME)

tmp_dir = os.path.join(HITL_TMP_DIR, "tmp")

COL_CONTENT = "content"
COL_FREQ = "freq"


def get_log_traceback(path: str):
    logging.info(f"Processing log file in {path}")
    # get log files in the path
    for fname in os.listdir(path):
        if fname.endswith(".log"):
            # found a log file
            fpath = os.path.join(path, fname)
            df = pd.DataFrame(columns=[COL_CONTENT, COL_FREQ])
            df = df.set_index(COL_CONTENT)

            with open(fpath) as file:
                content = ""
                for line in file:
                    # check if starts with YYYY-MM-DD
                    # or line starts with logging level
                    if (
                        re.match(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])", line)
                        or line.startswith("DEBUG")
                        or line.startswith("INFO")
                        or line.startswith("WARNING")
                        or line.startswith("ERROR")
                        or line.startswith("CRITICAL")
                    ):
                        # if the content exists & starts with trace back, append to df
                        if content and content.startswith("Traceback"):
                            if content not in df.index:
                                df.loc[content] = 0
                            df.loc[content] += 1
                        content = ""
                    else:
                        content += line

            if len(df) > 0:
                # Dedup based on content column and save
                df.to_csv(f"{fpath}.traceback.csv")


def data_gen_draft(file_path: str, folder_path: str):
    logging.info(f"Extracting from compressed file {file_path}")
    # unzip
    file = tarfile.open(file_path)
    file.extractall(folder_path)
    file.close()

    # process log
    get_log_traceback(path=folder_path)


for obj in bucket.objects.filter(Prefix=batch_prefix):
    # tempory file destination
    dest = os.path.join(tmp_dir, obj.key)
    # get folder path
    folder_path = dest[: dest.rindex("/")]
    os.makedirs(folder_path, exist_ok=True)
    logging.info("Retreiving %s from s3" % obj.key)

    try:
        bucket.download_file(obj.key, dest)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logging.error("The object does not exist.")
        else:
            raise

    # pass to data generator
    data_gen_draft(dest, folder_path)
