"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import boto3
import json
import logging
import os
import time

from droidlet.tools.hitl.job_listener import JobListener


HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
VISION_LISTENER_POLL_TIME = 30


session = boto3.Session(
         aws_access_key_id=AWS_ACCESS_KEY_ID,
         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
s3_resource = session.resource('s3')
s3_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
logging.basicConfig(level="INFO")


def vision_annotation(batch_id, timestamp, parsed): # test stub
    logging.info("vision_annotation test stub batch_id={batch_id} timestamp={timestamp} parsed={parsed}")


class VisionListener(JobListener):
    """
    This Data Generator is responsible for spinning up Annotation Jobs.

    Each Annotation Job is a HIT where turkers are asked to annotate the given command using the annotation tool.

    On a high level:
    - The input of this data generator is a single text command to be annotated
    - The output of this data generator is a (command, logical form) pair

    """

    def __init__(self, batch_id: int, command: str, cmd_id: int, timeout: float = -1) -> None:
        super(VisionListener, self).__init__(timeout)
        self._batch_id = batch_id
        self._command = command
        self._cmd_id = cmd_id

    def run(self) -> None:
    
        while not self.check_is_finished():
            try:
                s3_prefix = f"{self._batch_id}/interaction/"
                s3_objects = s3_resource.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)
                for file in s3_objects.get('Contents'):
                    name = file.key[len(s3_prefix)]
                    timestamp = name.split('/')[0]
                    body = file.get()['Body'].read()
                    parsed = json.loads(body)
                    vision_annotation(self._batch_id, timestamp, parsed)
                    # file.delete() # todo: uncomment when ready to run in production

                logging.info(f"Vision annotation completed")

            except:
                logging.info(
                    f"Vision annotation Job [{self._batch_id} terminated unexpectedly..."
                )

            if self.check_is_timeout():
                self.set_finished()
            time.sleep(VISION_LISTENER_POLL_TIME)


if __name__ == "__main__":
    vl = VisionListener(987, "destory the biggest house behind me", 1, 300)
    vl.run()
