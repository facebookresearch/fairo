#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import boto3
import argparse
from datetime import datetime
import csv
import sys
import yaml
import logging
import json
import time

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker

from droidlet.tools.hitl.vision_retrain.vision_labeling_jobs import VisionLabelingJob

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)
s3 = boto3.client('s3')
LABELING_JOB_TIMEOUT = 200

logging.basicConfig(level="INFO")


def main(opts) -> None:

    #Generate ID number from datettime
    id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logging.info(f"Batch ID: {id}")

    lj = VisionLabelingJob(id, opts.scene_length, opts.scene_height, opts.ground_depth, opts.max_num_shapes, opts.num_hits, opts.max_num_holes, LABELING_JOB_TIMEOUT)
    lj.run()

    #TODO How to turn off annotation parametrically?        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_length", type=int, default=18)
    parser.add_argument("--scene_height", type=int, default=13)
    parser.add_argument("--ground_depth", type=int, default=4)
    parser.add_argument("--max_num_shapes", type=int, default=4)
    parser.add_argument("--max_num_holes", type=int, default=3)
    parser.add_argument("--num_hits", type=int, default=1, help="Number of HITs to request")
    parser.add_argument("--mephisto_requester", type=str, default="ethancarlson_sandbox", help="Your Mephisto requester name")
    parser.add_argument("--annotate", action='store_true', help="Set to include annotate the scenes automatically")
    parser.add_argument("--labeling_timeout", type=int, default=200, help="Number of minutes before labeling job times out")
    parser.add_argument("--annotation_timeout", type=int, default=200, help="Number of minutes before annotation job times out")
    opts = parser.parse_args()
    main(opts)
