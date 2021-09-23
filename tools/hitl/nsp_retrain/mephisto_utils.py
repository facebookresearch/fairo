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


def check_run_status(run_id: int) -> None:
    db = LocalMephistoDB()
    units = db.find_units(task_run_id=run_id)
    units_num = len(units)
    completed_num = 0
    launched_num = 0
    assigned_num = 0
    accepted_num = 0
    for unit in units:
        if unit.db_status == "completed":
            completed_num += 1
        elif unit.db_status == "launched":
            launched_num += 1
        elif unit.db_status == "assigned":
            assigned_num += 1
        elif unit.db_status == "accepted":
            accepted_num += 1
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


def check_workers_in_allowlist(qual_name: str) -> None:
    db = LocalMephistoDB()
    all_workers = db.find_workers()
    cnt = 0
    for worker in all_workers:
        if worker.get_granted_qualification(qual_name):
            print(f"Worker# {worker.worker_name} is in allowlist")
            cnt += 1
    print(f"Total number of workers in db: {len(all_workers)} ")
    print(f"Number of workers in {qual_name} allowlist: {cnt}")


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


if __name__ == "__main__":
    check_run_status(81)
    backend_validation("/private/home/yuxuans/.hitl/20210824005202/parsed_turk_logs", "test")
    check_workers_in_allowlist("PILOT_ALLOWLIST_QUAL_0920_0")
    check_account_balance()
