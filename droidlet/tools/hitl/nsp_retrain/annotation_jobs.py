"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import os
import re
import shutil
import signal
import subprocess
import time

import boto3

from droidlet.tools.hitl.data_generator import DataGenerator

ANNOTATION_JOB_POLL_TIME = 5
S3_BUCKET_NAME = "craftassist"
S3_DIR = "turk_interactions_with_agent/turk"
S3_ROOT = "s3://craftassist/turk_interactions_with_agent/turk"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


class AnnotationJob(DataGenerator):
    def __init__(self, batch_id, timeout=-1):
        super(AnnotationJob, self).__init__(timeout)
        self._batch_id = batch_id

    def run(self):
        MTURK_AWS_ACCESS_KEY_ID = os.environ["MTURK_AWS_ACCESS_KEY_ID"]
        MTURK_AWS_SECRET_ACCESS_KEY = os.environ["MTURK_AWS_SECRET_ACCESS_KEY"]

        p = subprocess.Popen(
            [
                f"AWS_ACCESS_KEY_ID='{MTURK_AWS_ACCESS_KEY_ID}' AWS_SECRET_ACCESS_KEY='{MTURK_AWS_SECRET_ACCESS_KEY}' python ../../../../tools/annotation_tools/turk_with_s3/run_all_tasks.py"
            ],
            shell=True,
            preexec_fn=os.setsid,
        )

        # Keep running Mephisto until timeout or job finished
        while not self.check_is_timeout() and p.poll() is None:
            logging.debug(f"Annotation Job still running...")
            time.sleep(ANNOTATION_JOB_POLL_TIME)

        # if mturk job is still running after timeout, terminate it
        logging.info(f"Manually terminate turk job after timeout...")
        if p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(10)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        # upload annotated commands to S3
        combined_fname = "../../../../tools/annotation_tools/turk_with_s3/all_combined.txt"
        with open(combined_fname, "rb") as f:
            s3.upload_fileobj(
                f, f"{S3_BUCKET_NAME}", f"{S3_DIR}/{self._batch_id}/all_combined.txt"
            )

        self.set_finished()
