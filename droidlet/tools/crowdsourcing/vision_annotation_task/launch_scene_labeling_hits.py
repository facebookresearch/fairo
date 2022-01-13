#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Any
import subprocess
import boto3
import argparse
from datetime import datetime
import csv
import sys

s3 = boto3.client('s3')
SCENE_GEN_TIMEOUT = 30
LABELING_JOB_TIMEOUT = 18000


def main(opts) -> None:

    #Generate ID number from datettime
    id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    scene_filename = "scene" + id + ".json"
    hit_id_base = "hit" + id + "_"

    #Generate scenes
    scene_path = os.path.join(os.getcwd(), scene_filename)
    scene_gen_path = os.path.join(os.getcwd(), "../../../lowlevel/minecraft/small_scenes_with_shapes.py")
    scene_gen_cmd = "python3 " + \
        scene_gen_path + \
        " --NUM_SCENES=" + str(opts.num_hits) + \
        " --save_data_path=" + scene_path
    try:
        print(f"Starting scene generation script")
        scene_gen = subprocess.Popen(scene_gen_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except ValueError:
        print(f"Likely error: Popen called with invalid arguments")
        raise
    
    try:
        outs, _ = scene_gen.communicate(timeout=SCENE_GEN_TIMEOUT)
    except subprocess.TimeoutExpired:
        scene_gen.kill()
        outs, _ = scene_gen.communicate()
        print("Scene generation script timed out after {SCENE_GEN_TIMEOUT} seconds")
    print(f"Scene generation script output: \n{outs}")

    #Send scene file to S3
    upload_key = "pubr/scenes/" + scene_filename
    response = s3.upload_file(scene_path, 'craftassist', upload_key)
    if response: print(f"S3 upload response: {response}")

    #Populate data.csv with scene filename and hit IDs
    headers = ["scene_filename", "hit_id"]
    with open("labeling_data.csv", "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(headers)
        for i in range(opts.num_hits):
            csv_writer.writerow([scene_filename, (hit_id_base + str(i))])

    #Remove the local scene file
    os.remove(scene_path)

    #Launch via Mephisto
    job_launch_cmd = "python3 run_labeling_with_qual.py" + \
        " mephisto.provider.requester_name=" + opts.mephisto_requester + \
        " mephisto.architect.profile_name=mephisto-router-iam"
    try:
        print(f"Launching job with {opts.num_hits} HITs")
        job_launch = subprocess.Popen(job_launch_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, stdin=sys.stdin, text=True)
    except ValueError:
        print(f"Likely error: Popen called with invalid arguments")
        raise

    try:
        job_launch.wait(timeout=LABELING_JOB_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("Scene labeling job timed out after {LABELING_JOB_TIMEOUT} seconds")
    print(f"Job {id} complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=1, help="Number of HITs to request")
    parser.add_argument("--mephisto_requester", type=str, default="ethancarlson_sandbox", help="Your Mephisto requester name")
    opts = parser.parse_args()
    main(opts)
