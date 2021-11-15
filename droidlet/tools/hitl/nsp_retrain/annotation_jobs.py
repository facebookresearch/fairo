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

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
ANNOTATION_JOB_POLL_TIME = 30
ANNOTATION_PROCESS_TIMEOUT_DEFAULT = 120
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

logging.basicConfig(level="INFO")


class AnnotationJob(DataGenerator):
    """
    This Data Generator is responsible for spinning up Annotation Jobs.

    Each Annotation Job is a HIT where turkers are asked to annotate the given command using the annotation tool
    we've built previously which will generate a logical form for the annotated command after several steps.

    On a high level:
    - The input of this data generator is a single text command to be annotated
    - The output of this data generator is a (command, logical form) pair

    """

    def __init__(self, batch_id: int, command: str, cmd_id: int, timeout: float = -1) -> None:
        super(AnnotationJob, self).__init__(timeout)
        self._batch_id = batch_id
        self._command = command
        self._cmd_id = cmd_id

    def run(self) -> None:
        try:
            MTURK_AWS_ACCESS_KEY_ID = os.environ["MTURK_AWS_ACCESS_KEY_ID"]
            MTURK_AWS_SECRET_ACCESS_KEY = os.environ["MTURK_AWS_SECRET_ACCESS_KEY"]

            os.makedirs(f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}", exist_ok=True)
            with open(f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}/input.txt", "w+") as f:
                f.write(self._command)

            annotation_process_timeout = (
                ANNOTATION_PROCESS_TIMEOUT_DEFAULT
            )  # if self.get_remaining_time() < 0 else self.get_remaining_time() + 1
            p = subprocess.Popen(
                [
                        f"AWS_ACCESS_KEY_ID='{MTURK_AWS_ACCESS_KEY_ID}' AWS_SECRET_ACCESS_KEY='{MTURK_AWS_SECRET_ACCESS_KEY}' cd ../../../../tools/annotation_tools/turk_with_s3 && python run_all_tasks.py --default_write_dir={HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id} --timeout {annotation_process_timeout}"
                    ],
                #     f"AWS_ACCESS_KEY_ID='{MTURK_AWS_ACCESS_KEY_ID}' AWS_SECRET_ACCESS_KEY='{MTURK_AWS_SECRET_ACCESS_KEY}' cd ../../../../tools/annotation_tools/turk_with_s3 && python run_all_tasks.py --dev --default_write_dir={HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id} --timeout {annotation_process_timeout}"
                # ],
                shell=True,
                preexec_fn=os.setsid,
            )

            # Keep running Mephisto until timeout or job finished
            while not self.check_is_timeout() and p.poll() is None:
                logging.info(
                    f"Annotation Job [{self._batch_id}-{self._cmd_id}-{self._command}] still running...Remaining time: {self.get_remaining_time()}"
                )
                time.sleep(ANNOTATION_JOB_POLL_TIME)

            if p.poll() is None:
                # if mturk job is still running after timeout, terminate it
                logging.info(f"Manually terminate turk job after timeout...")
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
                time.sleep(10)
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)

            logging.info(
                f"Uploading annotated data {self._batch_id}/annotated/{self._cmd_id}_all_combined.txt to S3..."
            )
            # upload annotated commands to S3
            combined_fname = f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}/all_combined.txt"
            with open(combined_fname, "rb") as f:
                s3.upload_fileobj(
                    f,
                    f"{S3_BUCKET_NAME}",
                    f"{self._batch_id}/annotated/{self._cmd_id}_all_combined.txt",
                )
            logging.info(
                f"Uploading completed: {self._batch_id}/annotated/{self._cmd_id}_all_combined.txt"
            )
        except:
            logging.info(
                f"Annotation Job [{self._batch_id}-{self._cmd_id}-{self._command}] terminated unexpectedly..."
            )

        self.set_finished()


if __name__ == "__main__":
    aj = AnnotationJob(987, "destory the biggest house behind me", 1, 300)
    aj.run()
