#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import logging
import boto3

from mephisto.data_model.worker import Worker
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser

from droidlet_static_html_task.pilot_config import PILOT_ALLOWLIST_QUAL_NAME as interaction_whitelist
from droidlet_static_html_task.pilot_config import PILOT_BLOCK_QUAL_NAME as interaction_blacklist
from vision_annotation_task.pilot_config import PILOT_ALLOWLIST_QUAL_NAME as vision_annotation_whitelist
from vision_annotation_task.pilot_config import PILOT_BLOCK_QUAL_NAME as vision_annotation_blacklist

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

s3 = boto3.client('s3')


def importAllowlist(bucket: str, filename: str):
    with open('allowlist.txt', 'wb') as f:
        s3.download_fileobj(bucket, filename, f)
        allowlist = [line.strip() for line in f.readlines()]
    logging.info(f"Updated whitelist downloaded successfully")

    os.remove("allowlist.txt")

    return allowlist


def addQual(worker, whitelist, blacklist):
    # Block every worker from working on this pilot task for second time
    try:
        db.make_qualification(blacklist)
    except:
        pass
    else:
        logging.debug(f"{blacklist} qualification not exists, so create one")
    worker.grant_qualification(blacklist, 1)
    
    # Workers who pass the validation will be put into an allowlist
    try:
        db.make_qualification(whitelist)
    except:
        pass
    else:
        logging.debug(f"{whitelist} qualification not exists, so create one")
    worker.grant_qualification(whitelist, 1)
    logging.info(f"Worker {worker.worker_name} passed the pilot task, put him/her into allowlist")

    return


def addWorkerToList(allowlist, allow_qual, block_qual):
    for turker in allowlist:
        #First add the worker to the database, or retrieve them if they already exist
        try:
            db_id = db.new_worker(turker, 'mturk')
            worker = Worker.get(db, db_id)
        except:
            worker = db.find_workers(turker, 'mturk')[0]

        #Then add them to the qualification
        addQual(worker, allow_qual, block_qual)

        #Check to make sure the qualification was added successfully
        if not worker.is_qualified(allow_qual):
            logging.info(f"!!! {worker} not successfully qualified, debug")
        else:
            logging.info(f"{worker} successfully qualified")


def main(bucket, int_list, vis_list, nlu_list):
    # Pull shared lists from S3
    s3_int_allowlist = importAllowlist(bucket, int_list)
    s3_vis_allowlist = importAllowlist(bucket, vis_list)
    s3_nlu_allowlist = importAllowlist(bucket, nlu_list)

    # TODO block lists should be sync'd as well
    

    # Check against local lists

    
    # Send new local workers to S3

    
    # Add local qual to new workers on S3


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket where allowlists are stored")
    parser.add_argument("--interaction_list_name", type=str, help="Name of interaction allowlist on S3")
    parser.add_argument("--vis_annotation_list_name", type=str, help="Name of vision annotation allowlist on S3")
    parser.add_argument("--nlu_annotation_list_name", type=str, help="Name of nlu annotation allowlist on S3")
    opts = parser.parse_args()
    main(opts.s3_bucket, opts.interaction_list_name, opts.vis_annotation_list_name, opts.nlu_annotation_list_name)