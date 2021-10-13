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
    def __init__(self, batch_id, command, cmd_id, timeout=-1):
        super(AnnotationJob, self).__init__(timeout)
        self._batch_id = batch_id
        self._command = command
        self._cmd_id = cmd_id

    def run(self):
        try:
            MTURK_AWS_ACCESS_KEY_ID = os.environ["MTURK_AWS_ACCESS_KEY_ID"]
            MTURK_AWS_SECRET_ACCESS_KEY = os.environ["MTURK_AWS_SECRET_ACCESS_KEY"]

            os.makedirs(f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}", exist_ok=True)
            with open(f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}/input.txt", "w+") as f:
                f.write(self._command)

            annotation_process_timeout = ANNOTATION_PROCESS_TIMEOUT_DEFAULT #if self.get_remaining_time() < 0 else self.get_remaining_time() + 1
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
                logging.info(f"Annotation Job [{self._batch_id}-{self._cmd_id}-{self._command}] still running...Remaining time: {self.get_remaining_time()}")
                time.sleep(ANNOTATION_JOB_POLL_TIME)

            if p.poll() is None:
                # if mturk job is still running after timeout, terminate it
                logging.info(f"Manually terminate turk job after timeout...")
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
                time.sleep(10)
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)

            logging.info(f"Uploading annotated data {self._batch_id}/annotated/{self._cmd_id}_all_combined.txt to S3...")
            # upload annotated commands to S3
            combined_fname = f"{HITL_TMP_DIR}/{self._batch_id}/{self._cmd_id}/all_combined.txt"
            with open(combined_fname, "rb") as f:
                s3.upload_fileobj(
                    f, f"{S3_BUCKET_NAME}", f"{self._batch_id}/annotated/{self._cmd_id}_all_combined.txt"
                )
            logging.info(f"Uploading completed: {self._batch_id}/annotated/{self._cmd_id}_all_combined.txt")
        except:
            logging.info(f"Annotation Job [{self._batch_id}-{self._cmd_id}-{self._command}] terminated unexpectedly...")

        self.set_finished()

def delete_mturk_hits():
    import os
    import boto3
    from datetime import datetime

    access_key = os.getenv("MTURK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MTURK_AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("MTURK_AWS_REGION", default="us-east-1")

    MTURK_URL = "https://mturk-requester-sandbox.{}.amazonaws.com".format(aws_region)

    mturk = boto3.client(
        "mturk",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
        endpoint_url=MTURK_URL,
    )

    all_hits = mturk.list_hits()["HITs"]
    hit_ids = [item["HITId"] for item in all_hits]
    # This is slow but there's no better way to get the status of pending HITs
    for hit_id in hit_ids:
        # Get HIT status
        status = mturk.get_hit(HITId=hit_id)["HIT"]["HITStatus"]
        try:
            response = mturk.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1))
            mturk.delete_hit(HITId=hit_id)
        except:
            pass
        print(f"Hit {hit_id}, status: {status}")

if __name__ == "__main__":
    aj = AnnotationJob(987, "destory the biggest house behind me", 1, 300)
    aj.run()
    # delete_mturk_hits()

    

