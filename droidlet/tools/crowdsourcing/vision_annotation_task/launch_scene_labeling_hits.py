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

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker

from droidlet.tools.hitl.vision_retrain.vision_annotation_jobs import VisionAnnotationJob

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)
s3 = boto3.client('s3')
SCENE_GEN_TIMEOUT = 30
LABELING_JOB_TIMEOUT = 18000

logging.basicConfig(level="INFO")


def main(opts) -> None:

    #Generate ID number from datettime
    id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logging.info(f"Batch ID: {id}")

    #Generate scenes
    scene_path = os.path.join(os.getcwd(), "server_files/extra_refs/scene_list.json")
    scene_gen_path = os.path.join(os.getcwd(), "../../../lowlevel/minecraft/small_scenes_with_shapes.py")
    scene_gen_cmd = "python3 " + \
        scene_gen_path + \
        " --SL=" + str(opts.scene_length) + \
        " --H=" + str(opts.scene_height) + \
        " --GROUND_DEPTH=" + str(opts.ground_depth) + \
        " --MAX_NUM_SHAPES=" + str(opts.max_num_shapes) + \
        " --NUM_SCENES=" + str(opts.num_hits) + \
        " --save_data_path=" + scene_path
    try:
        logging.info(f"Starting scene generation script")
        scene_gen = subprocess.Popen(scene_gen_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    except ValueError:
        logging.info(f"Likely error: Popen called with invalid arguments")
        raise
    
    try:
        scene_gen.wait(timeout=SCENE_GEN_TIMEOUT)
    except subprocess.TimeoutExpired:
        scene_gen.kill()
        logging.info("Scene generation script timed out after {SCENE_GEN_TIMEOUT} seconds")

    #Send scene file to S3 for future annotation
    upload_key = id + "/vision_annotation/scene_list.json"
    response = s3.upload_file(scene_path, 'droidlet-hitl', upload_key)
    if response: logging.info(f"S3 upload response: {response}")

    #Populate data.csv with scene filename and indeces
    with open("labeling_data.csv", "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["scene_filename", "scene_idx"])
        for i in range(opts.num_hits):
            csv_writer.writerow(["scene_list.json", str(i)])

    # Edit Mephisto config file task name
    with open("conf/labeling.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        task_name = "ca-vis-label" + str(id)
        config["mephisto"]["task"]["task_name"] = task_name
    logging.info(f"Updating Mephisto config file to have task_name: {task_name}")
    with open("conf/labeling.yaml", "w") as stream:
        stream.write("#@package _global_\n")
        yaml.dump(config, stream)

    #Launch via Mephisto
    job_launch_cmd = "python3 run_labeling_with_qual.py" + \
        " mephisto.provider.requester_name=" + opts.mephisto_requester + \
        " mephisto.architect.profile_name=mephisto-router-iam"
    try:
        logging.info(f"Launching job with {opts.num_hits} HITs")
        job_launch = subprocess.Popen(job_launch_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, stdin=sys.stdin, text=True)
    except ValueError:
        logging.info(f"Likely error: Popen called with invalid arguments")
        raise

    try:
        job_launch.wait(timeout=LABELING_JOB_TIMEOUT)
    except subprocess.TimeoutExpired:
        job_launch.kill()
        logging.info("Scene labeling job timed out after {LABELING_JOB_TIMEOUT} seconds")
    
    # Pull results from local DB    
    results_csv = id + ".csv"
    with open(results_csv, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["scene_filename", "scene_idx", "worker_name", "object", "location"])

        units = mephisto_data_browser.get_units_for_task_name(task_name)
        scene_list = []
        label_list = []
        for unit in units:
            data = mephisto_data_browser.get_data_from_unit(unit)
            worker_name = Worker(db, data["worker_id"]).worker_name
            object = data["data"]["outputs"]["object"]
            label_list.append(object)
            location = data["data"]["outputs"]["location"]
            scene_idx = data["data"]["outputs"]["scene_idx"]
            scene_list.append(scene_idx)
            csv_writer.writerow(["scene_list.json", scene_idx, worker_name, object, location])

    # Upload results to S3
    upload_key = id + "/vision_labeling_results/" + results_csv
    logging.info(f"Uploading job results to S3: {upload_key}")
    response = s3.upload_file(results_csv, 'droidlet-hitl', upload_key)
    if response: logging.info("S3 response: " + response)

    # Remove local temp files
    os.remove(results_csv)
    os.remove(scene_path)

    logging.info(f"Labeling job {id} complete")
    
    if opts.annotate:
        logging.info(f"Launching corresponding annotation job")

        aj = VisionAnnotationJob(id, scene_list, label_list, 300)
        aj.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_length", type=int, default=20)
    parser.add_argument("--scene_height", type=int, default=14)
    parser.add_argument("--ground_depth", type=int, default=3)
    parser.add_argument("--max_num_shapes", type=int, default=4)
    parser.add_argument("--num_hits", type=int, default=1, help="Number of HITs to request")
    parser.add_argument("--mephisto_requester", type=str, default="ethancarlson_sandbox", help="Your Mephisto requester name")
    parser.add_argument("--annotate", action='store_true', help="Set to include annotate the scenes automatically")
    opts = parser.parse_args()
    os.environ["MEPHISTO_REQUESTER"] = opts.mephisto_requester
    main(opts)
