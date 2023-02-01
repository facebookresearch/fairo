#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" 
This script syncs Mephisto (MTurk) allowlists and blocklists between the
local Mephisto DB and shared lists (.txt files) in an S3 bucket.

Currently implemented are the interaction job and vision annotation job
lists, but the structure is extendable to future qualifications as well.
"""

import argparse
import os
import logging
import boto3
import copy

from mephisto.data_model.worker import Worker
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser

from droidlet_static_html_task.pilot_config import (
    PILOT_ALLOWLIST_QUAL_NAME as interaction_whitelist,
)
from droidlet_static_html_task.pilot_config import PILOT_BLOCK_QUAL_NAME as interaction_blacklist
from droidlet_static_html_task.pilot_config import SOFTBLOCK_QUAL_NAME as interaction_softblock
from vision_annotation_task.pilot_config import (
    PILOT_ALLOWLIST_QUAL_NAME as vision_annotation_whitelist,
)
from vision_annotation_task.pilot_config import (
    PILOT_BLOCK_QUAL_NAME as vision_annotation_blacklist,
)
from vision_annotation_task.pilot_config import SOFTBLOCK_QUAL_NAME as vision_softblock

qual_dict = {
    "interaction": {
        "allow": interaction_whitelist,
        "block": interaction_blacklist,
        "softblock": interaction_softblock,
    },
    "vision_annotation": {
        "allow": vision_annotation_whitelist,
        "block": vision_annotation_blacklist,
        "softblock": vision_softblock,
    },
}

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

s3 = boto3.client("s3")

logging.basicConfig(level="INFO")


def import_s3_lists(bucket: str):
    # Assumed S3 allowlist key example: (bucket)/interaction/allow.txt

    output_dict = copy.deepcopy(qual_dict)

    for task in output_dict.keys():
        for list_type in output_dict[task].keys():
            key = f"{task}/{list_type}.txt"
            try:
                with open("list.txt", "wb") as f:
                    s3.download_fileobj(bucket, key, f)
                logging.info(f"{task} {list_type}list downloaded successfully")
                with open("list.txt", "r") as f:
                    output_dict[task][list_type] = [line.strip() for line in f.readlines()]
            except:
                logging.info(
                    f"{task} {list_type}list not found on S3, creating new S3 {list_type}list"
                )
                output_dict[task][list_type] = []

    os.remove("list.txt")
    return output_dict


def add_workers_to_quals(add_list: list, qual: str):
    for turker in add_list:
        # First add the worker to the database, or retrieve them if they already exist
        try:
            db_id = db.new_worker(turker, "mturk")
            worker = Worker.get(db, db_id)
        except:
            worker = db.find_workers(turker, "mturk")[0]

        # Add the worker to the relevant list
        try:
            db.make_qualification(qual)
        except:
            pass
        else:
            logging.debug(f"{qual} qualification not exists, so create one")
        worker.grant_qualification(qual, 1)

        # Check to make sure the qualification was added successfully
        if not worker.is_qualified(qual):
            logging.info(f"!!! {worker} not successfully qualified, debug")
        else:
            logging.info(f"Worker {worker.worker_name} added to list {qual}")


def pull_local_lists():
    # Pull the qual lists from local Mephisto DB into a formatted dict

    output_dict = copy.deepcopy(qual_dict)

    logging.info(f"Retrieving qualification lists from local Mephisto DB")
    for task in output_dict.keys():
        for list_type in output_dict[task].keys():
            # If syncing for the first time, qualifications may not yet exist
            try:
                logging.info(f"attempting to make qualification: {qual_dict[task][list_type]}")
                db.make_qualification(qual_dict[task][list_type])
            except:
                logging.info(f"Qualification {qual_dict[task][list_type]} already exists")
                pass
            qual_list = mephisto_data_browser.get_workers_with_qualification(
                qual_dict[task][list_type]
            )
            output_dict[task][list_type] = [worker.worker_name.strip("\n") for worker in qual_list]

    return output_dict


def compare_qual_lists(s3_lists: dict, local_lists: dict):
    # Compare two dicts of lists representing the local and S3 states, return a dict with the differences

    diff_dict = copy.deepcopy(qual_dict)

    logging.info(f"Comparing qualification lists and checking for differences")
    for t in diff_dict.keys():
        for l in diff_dict[t].keys():
            diff_dict[t][l] = {}
            diff_dict[t][l]["s3_exclusive"] = [
                x for x in s3_lists[t][l] if x not in local_lists[t][l]
            ]
            diff_dict[t][l]["local_exclusive"] = [
                x for x in local_lists[t][l] if x not in s3_lists[t][l]
            ]

    return diff_dict


def update_lists(bucket: str, diff_dict: dict):
    # Iterate through the differences between local and S3 lists and update both to be in sync

    for t in diff_dict.keys():
        for l in diff_dict[t].keys():
            for e in diff_dict[t][l].keys():
                if e == "s3_exclusive" and len(diff_dict[t][l][e]) > 0:
                    add_workers_to_quals(diff_dict[t][l][e], qual_dict[t][l])

                elif e == "local_exclusive" and len(diff_dict[t][l][e]) > 0:
                    logging.info(
                        f"Writing new workers to {t} {l} shared list on S3: {diff_dict[t][l][e]}"
                    )

                    filename = l + ".txt"
                    with open(filename, "w") as f:
                        for line in diff_dict[t][l][e]:
                            f.write(line.strip("\n") + "\n")

                    upload_key = t + "/" + filename
                    s3.upload_file(filename, bucket, upload_key)
                    logging.info(f"S3 upload succeeded")

                    os.remove(filename)

                else:
                    logging.info(f"No {e} workers on {t} {l} list, no update performed")

    return


def main(bucket: str):
    # Pull shared lists from S3 and local qual lists
    s3_list_dict = import_s3_lists(bucket)
    local_list_dict = pull_local_lists()

    # Compare them for differences
    diff_dict = compare_qual_lists(s3_list_dict, local_list_dict)

    # Update local and s3 lists to match
    update_lists(bucket, diff_dict)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_bucket", type=str, required=True, help="S3 bucket where allowlists are stored"
    )
    opts = parser.parse_args()
    main(opts.s3_bucket)
