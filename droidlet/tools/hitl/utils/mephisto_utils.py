"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import logging
import os

import boto3
import pandas as pd

from typing import List

from droidlet.dialog.load_datasets import get_safety_words

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.abstractions.providers.mturk.mturk_utils import get_hit

from mephisto.tools.data_browser import DataBrowser


NSP_LOG_FNAME = "nsp_outputs.csv"
METADATA_FNAME = "job_metadata.json"

MIN_INTERACTION_CMD_NUM = 5
MIN_CMD_AVG_LEN = 2
KEY_WORD_RATIO = 0.1

KEYWORD_LIST = [
    "build",
    "destroy",
    "dance",
    "get",
    "tag",
    "dig",
    "copy",
    "undo",
    "fill",
    "spawn",
    "answer",
    "stop",
    "resume",
    "come",
    "go",
    "make",
]
SAFETY_WORDS = get_safety_words()

logger = logging.getLogger()
logger.setLevel("INFO")


def check_run_status(run_id: int, mturk_with_qual_list) -> None:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    units_num = len(units)
    completed_num = 0
    launched_num = 0
    assigned_num = 0
    accepted_num = 0
    completed_units = []
    for unit in units:
        if unit.db_status == "completed":
            completed_num += 1
            completed_units.append(unit)
        elif unit.db_status == "launched":
            launched_num += 1
        elif unit.db_status == "assigned":
            assigned_num += 1
        elif unit.db_status == "accepted":
            accepted_num += 1
    print(f"Total unit num: {units_num}")
    print(
        f"Total HIT num: {units_num}\tCompleted HIT num: {completed_num}\tCompleted rate: {completed_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tLaunched HIT num: {launched_num}\tLaunched rate: {launched_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tAssigned HIT num: {assigned_num}\tAssigned rate: {assigned_num / units_num * 100}%"
    )
    print(
        f"Total HIT num: {units_num}\tAccepted HIT num: {accepted_num}\tAccepted rate: {accepted_num / units_num * 100}%"
    )

    data_browser = DataBrowser(db=db)
    total_time_completed_in_min = 0
    total_cnt = 0
    passed_time = 0
    passed_cnt = 0
    qual_name = "PILOT_ALLOWLIST_QUAL_0920_0"
    turkers_with_mturk_qual_cnt = 0
    turker_set = set()
    for unit in completed_units:
        data = data_browser.get_data_from_unit(unit)
        duration = (
            data["data"]["times"]["task_end"] - data["data"]["times"]["task_start"]
        ) / 60  # in minutes
        total_time_completed_in_min += duration
        total_cnt += 1
        worker_name = db.get_worker(worker_id=unit.worker_id)["worker_name"]
        print(f"Examine run unit completed by {worker_name}")
        turker_set.add(worker_name)
        if worker_name not in mturk_with_qual_list:
            print(f"This Mephisto worker {worker_name} is not in the mturk allowlist")
        else:
            turkers_with_mturk_qual_cnt += 1
        # print(f"Worker {worker_name} works on HIT {run_id}")
        worker = db.find_workers(worker_name=worker_name)[0]
        if worker.get_granted_qualification(qual_name):
            passed_time += duration
            passed_cnt += 1
        else:
            print(f"This Mephisto worker {worker_name} is not in the Mephisto allowlist")
    print(turker_set)
    print(f"Identical turker num: {len(set(list(turker_set)))}")
    print(
        f"For mephisto/mturk debug: total num: {total_cnt}, # who pass mturk qual: {turkers_with_mturk_qual_cnt}"
    )
    print(
        f"Total completed HITS\t\t{total_cnt}\tavg time spent\t{total_time_completed_in_min / total_cnt} mins"
    )
    print(
        f"HITS passed qualification\t{passed_cnt}\tavg time spent\t{passed_time / passed_cnt} mins"
    )
    print(
        f"HITS failed qualification\t{total_cnt - passed_cnt}\tavg time spent\t{(total_time_completed_in_min - passed_time) / (total_cnt - passed_cnt)} mins"
    )


def check_workers_in_allowlist(qual_name: str, turker_list=None) -> None:
    db = LocalMephistoDB()
    all_workers = db.find_workers()
    cnt = 0
    allowed_workers = []
    for worker in all_workers:
        if worker.get_granted_qualification(qual_name):
            allowed_workers.append(worker.worker_name)
            print(f"Worker# {worker.worker_name} is in allowlist {qual_name}")
            if turker_list is not None:
                if worker.worker_name not in turker_list:
                    print("Discrepency between mephisto and mturk")
            cnt += 1
    print(f"Total number of workers in db: {len(all_workers)} ")
    print(f"Number of workers in {qual_name} allowlist: {cnt}")
    return allowed_workers


def check_all_qual() -> None:
    db = LocalMephistoDB()
    quals = db.find_qualifications()
    for qual in quals:
        print(qual.qualification_name)


def check_account_balance(
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    aws_default_region: str = None,
) -> None:
    aws_access_key_id = aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = aws_secret_access_key or os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_default_region = aws_default_region or os.environ["AWS_DEFAULT_REGION"]
    mturk_url = "https://mturk-requester.{}.amazonaws.com".format(aws_default_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
        endpoint_url=mturk_url,
    )
    print(f"Account balance: {mturk.get_account_balance()}")


def check_mturk_qual_turkers(
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    aws_default_region: str = None,
    qual_id: str = None,
) -> None:
    aws_access_key_id = aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = aws_secret_access_key or os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_default_region = aws_default_region or os.environ["AWS_DEFAULT_REGION"]
    mturk_url = "https://mturk-requester.{}.amazonaws.com".format(aws_default_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
        endpoint_url=mturk_url,
    )
    results = mturk.list_workers_with_qualification_type(QualificationTypeId=qual_id)
    partial_qual_list = results["Qualifications"]
    next_token = results["NextToken"]
    full_worker_list = []
    while next_token != None:
        for qual in partial_qual_list:
            full_worker_list.append(qual["WorkerId"])
        results = mturk.list_workers_with_qualification_type(
            QualificationTypeId=qual_id, NextToken=next_token
        )
        partial_qual_list = results["Qualifications"]
        next_token = None if not "NextToken" in results else results["NextToken"]
    return full_worker_list


def validate_commands(commands: List[str]) -> bool:
    # filter empty commands
    filtered_commands = [x for x in commands if x != ""]

    # 1: Check that the number of commands >= 4
    if len(filtered_commands) < MIN_INTERACTION_CMD_NUM:
        return False

    # 2: Length check: Check that the average number of words in commands >= 2
    commands_split = [x.lower().split(" ") for x in filtered_commands]
    avg_words_in_commands = sum(map(len, commands_split)) / len(commands_split)
    if avg_words_in_commands < MIN_CMD_AVG_LEN:
        return False

    # 3: Keyword matching: Check that at least 2 keywords appear in the commands
    occurrence = sum(
        [
            True if keyword in command.lower() else False
            for command in filtered_commands
            for keyword in KEYWORD_LIST
        ]
    )
    if occurrence < len(filtered_commands) * KEY_WORD_RATIO:
        return False

    # 4: Safety check
    if any([len(set(cmd) & SAFETY_WORDS) > 0 for cmd in commands_split]):
        return False

    return True


def backend_validation(turk_dir_root: str, qual_name: str) -> None:
    """
    Do backend validation of turkers interaction log and soft block those
    who failed the validation by removing them from the allowlist.

    :param str turk_dir_root: root directory contains all turkers' logs
    :param str qual_name: qualification name of the allowlist
    """
    db = LocalMephistoDB()
    for turk_dir in os.listdir(turk_dir_root):
        nsp_log = os.path.join(os.path.join(turk_dir_root, turk_dir), NSP_LOG_FNAME)
        commands = pd.read_csv(nsp_log, delimiter="|")["command"].tolist()
        meta_file = os.path.join(os.path.join(turk_dir_root, turk_dir), METADATA_FNAME)
        with open(meta_file, "r+") as f:
            turker_metadata = json.load(f)

        if validate_commands(commands) is not True:
            workers = db.find_workers(worker_name=turker_metadata["turk_worker_id"])
            if len(workers) != 1:
                logging.warning(
                    f"{len(workers)} is found with name: {turker_metadata['turk_worker_id']}, it doesn't seem to be right..."
                )
            else:
                worker = workers[0]
                if worker.is_qualified(qual_name) and worker.revoke_qualification(qual_name):
                    logging.info(
                        f"Worker [{worker.worker_name}] failed backend validation, revoke qualification [{qual_name}] and soft-block on future HITs."
                    )


def check_mturk_datastore():
    db = LocalMephistoDB()
    turk_db = db.get_datastore_for_provider("mturk")
    mapping = turk_db.get_qualification_mapping("PILOT_ALLOWLIST_QUAL_0920_0")
    print(mapping["mturk_qualification_id"])


def grant_qual_to_turker(
    qual_id,
    worker_id,
    aws_access_key_id: str = None,
    aws_secret_access_key: str = None,
    aws_default_region: str = None,
):
    aws_access_key_id = aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = aws_secret_access_key or os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_default_region = aws_default_region or os.environ["AWS_DEFAULT_REGION"]
    mturk_url = "https://mturk-requester.{}.amazonaws.com".format(aws_default_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
        endpoint_url=mturk_url,
    )

    mturk.associate_qualification_with_worker(QualificationTypeId=qual_id, WorkerId=worker_id)


if __name__ == "__main__":
    QUAL_ID = ""
    workers_with_qual = check_mturk_qual_turkers(qual_id=QUAL_ID)
    check_run_status(205, mturk_with_qual_list=workers_with_qual)
    check_workers_in_allowlist("PILOT_ALLOWLIST_QUAL_0920_0")
