#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import logging
import boto3
import copy

from mephisto.data_model.worker import Worker
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser

from droidlet_static_html_task.pilot_config import PILOT_ALLOWLIST_QUAL_NAME as interaction_whitelist
from droidlet_static_html_task.pilot_config import PILOT_BLOCK_QUAL_NAME as interaction_blacklist
from vision_annotation_task.pilot_config import PILOT_ALLOWLIST_QUAL_NAME as vision_annotation_whitelist
from vision_annotation_task.pilot_config import PILOT_BLOCK_QUAL_NAME as vision_annotation_blacklist

qual_dict = {"interaction": {
                "allow": interaction_whitelist, 
                "block": interaction_blacklist }, 
            "vision_annotation": {
                "allow": vision_annotation_whitelist, 
                "block": vision_annotation_blacklist } }

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

s3 = boto3.client('s3')


def import_s3_lists(bucket: str):
    output_dict = copy.copy(qual_dict)
    
    for folder in output_dict.keys():
        key = folder + "/allow.txt"
        with open('list.txt', 'wb') as f:
            s3.download_fileobj(bucket, key, f)
            output_dict[folder]["allow"] = [line.strip() for line in f.readlines()]

        key = folder + "/block.txt"
        with open('list.txt', 'wb') as f:
            s3.download_fileobj(bucket, key, f)
            output_dict[folder]["block"] = [line.strip() for line in f.readlines()]

        logging.info(f"{folder} whitelist and blacklist downloaded successfully")

    os.remove("list.txt")
    return output_dict


def add_workers_to_quals(add_list: list, qual: str):
    for turker in add_list:
        #First add the worker to the database, or retrieve them if they already exist
        try:
            db_id = db.new_worker(turker, 'mturk')
            worker = Worker.get(db, db_id)
        except:
            worker = db.find_workers(turker, 'mturk')[0]
        
        # Add the worker to the relevant list
        try:
            db.make_qualification(qual)
        except:
            pass
        else:
            logging.debug(f"{qual} qualification not exists, so create one")
        worker.grant_qualification(qual, 1)

        #Check to make sure the qualification was added successfully
        if not worker.is_qualified(qual):
            logging.info(f"!!! {worker} not successfully qualified, debug")
        else:
            logging.info(f"Worker {worker.worker_name} added to list {qual}")


def pull_local_lists():
    output_dict = copy.copy(qual_dict)

    logging.info(f"Retrieving qualification lists from local Mephisto DB")
    for task in output_dict.keys():
        whitelist = mephisto_data_browser.get_workers_with_qualification(qual_dict[task]["allow"])
        output_dict[task]["allow"] = [worker.worker_name for worker in whitelist]
        blacklist = mephisto_data_browser.get_workers_with_qualification(qual_dict[task]["block"])
        output_dict[task]["block"] = [worker.worker_name for worker in blacklist]

    return output_dict


def compare_qual_lists(s3_lists: dict, local_lists: dict):
    diff_dict = copy.copy(qual_dict)

    for t in diff_dict.keys():
        for l in diff_dict[t].keys():
            diff_dict[t][l]["s3_exclusive"] = [x for x in s3_lists[t][l] if x not in local_lists[t][l]]
            diff_dict[t][l]["local_exclusive"] = [x for x in local_lists[t][l] if x not in s3_lists[t][l]]

    return diff_dict


def update_lists(bucket:str, diff_dict: dict):

    for t in diff_dict.keys():
        for l in diff_dict[t].keys():
            for e in diff_dict[t][l].keys():

                if e == "s3_exclusive" and len(diff_dict[t][l][e]) > 0:
                    logging.info(f"Writing new workers to shared lists on S3: {diff_dict[t][l][e]}")

                    filename = l + ".txt"
                    with open(filename, "w") as f:
                        f.write(line + '\n' for line in diff_dict[t][l][e])
                    
                    upload_key = t + "/" + filename
                    s3.upload_file(filename, bucket, upload_key)
                    logging.info(f"S3 upload succeeded")
                    
                    os.remove(filename)

                elif e == "local_exclusive" and len(diff_dict[t][l][e]) > 0:
                    add_workers_to_quals(diff_dict[t][l][e], qual_dict[t][l])

    return


def main(bucket):
    # Pull shared lists from S3 and local qual lists
    s3_list_dict = import_s3_lists(bucket)
    local_list_dict = pull_local_lists()

    # Compare them for differences
    diff_dict = compare_qual_lists(s3_list_dict, local_list_dict)
    
    # Update local and s3 lists to match
    update_lists(diff_dict)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket where allowlists are stored")
    opts = parser.parse_args()
    main(opts.s3_bucket)