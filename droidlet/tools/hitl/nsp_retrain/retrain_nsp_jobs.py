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

from typing import List

import boto3
import botocore

from annotation_jobs import AnnotationJob
from allocate_instances import allocate_instances
from hitl_utils import generate_batch_id, deregister_dashboard_subdomain
from process_s3_logs import read_s3_bucket, read_turk_logs

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner

INTERACTION_JOB_POLL_TIME = 5
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "craftassist"
S3_DIR = "turk_interactions_with_agent/turk"
S3_ROOT = "s3://craftassist/turk_interactions_with_agent/turk"
NSP_OUTPUT_FNAME = "nsp_outputs"

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


class InteractionJob(DataGenerator):
    def __init__(self, instance_num, timeout=-1):
        super(InteractionJob, self).__init__(timeout)
        self._instance_num = instance_num
        self._batch_id = generate_batch_id()

    def run(self):
        batch_id = self._batch_id

        # allocate AWS ECS instances and register DNS records
        logging.info(f"Allocate AWS ECS instances and register DNS records...")
        allocate_instances(self._instance_num, batch_id)

        # run Mephisto to spin up & monitor turk jobs
        logging.info(f"Start running Mephisto...")
        MEPHISTO_AWS_ACCESS_KEY_ID = os.environ["MEPHISTO_AWS_ACCESS_KEY_ID"]
        MEPHISTO_AWS_SECRET_ACCESS_KEY = os.environ["MEPHISTO_AWS_SECRET_ACCESS_KEY"]
        MEPHISTO_REQUESTER = os.environ["MEPHISTO_REQUESTER"]
        p = subprocess.Popen(
            [
                f"echo -ne '\n' |  AWS_ACCESS_KEY_ID='{MEPHISTO_AWS_ACCESS_KEY_ID}' AWS_SECRET_ACCESS_KEY='{MEPHISTO_AWS_SECRET_ACCESS_KEY}' python ../../crowdsourcing/droidlet_static_html_task/static_run_with_onboarding.py mephisto.provider.requester_name={MEPHISTO_REQUESTER}"
            ],
            shell=True,
            preexec_fn=os.setsid,
        )

        # Keep running Mephisto until timeout or job finished
        while not self.check_is_timeout() and p.poll() is None:
            logging.debug(f"Interaction Job still running...")
            time.sleep(INTERACTION_JOB_POLL_TIME)

        # if mephisto is still running after job timeout, terminate it
        logging.info(f"Manually terminate Mephisto after timeout...")
        if p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(10)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        # Deregister DNS records
        logging.info(f"Deregister DNS records...")
        deregister_dashboard_subdomain(batch_id)

        batch_id = "20210824005202"
        self.process_s3_logs(batch_id)

        self.set_finished()

    def process_s3_logs(self, batch_id):
        s3_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/turk_logs")
        parsed_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/parsed_turk_logs")
        os.makedirs(s3_logs_dir, exist_ok=True)
        os.makedirs(parsed_logs_dir, exist_ok=True)

        rc = subprocess.call([f"aws s3 sync {S3_ROOT}/{batch_id} {s3_logs_dir}"], shell=True)

        read_s3_bucket(s3_logs_dir, parsed_logs_dir)
        command_list = read_turk_logs(parsed_logs_dir, NSP_OUTPUT_FNAME)
        logging.info(f"command list from interactions: {command_list}")

        logging.info(f"Uploading command list to S3...")
        content = "\n".join(command_list)
        s3.Object(f"{S3_BUCKET_NAME}", f"{S3_DIR}/{batch_id}/collected_commands").put(Body=content)

    def get_batch_id(self):
        return self._batch_id


class InteractionLogListener(JobListener):
    def __init__(self, batch_id):
        super(InteractionLogListener, self).__init__()
        self._batch_id = batch_id

    def run(self, runner):
        batch_id = self._batch_id
        while not self.check_is_finished():
            finished = False
            try:
                s3.Object(f"{S3_BUCKET_NAME}", f"{S3_DIR}/{batch_id}/collected_commands").load()
            except botocore.exceptions.ClientError as e:
                logging.info(f"Collected commands not found yet...")
            else:
                # Download command file from S3 and spin up annotation job
                input_path = "../../../../tools/annotation_tools/turk_with_s3/input.txt"
                s3.download_file(
                    f"{S3_BUCKET_NAME}", f"{S3_DIR}/{batch_id}/collected_commands", input_path
                )
                annotation_job = AnnotationJob(batch_id)
                runner.register_data_generators([annotation_job])

            if self.check_parent_finished():
                self.set_finished()
