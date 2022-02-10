"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import boto3
import json
import logging
import os
import time
import botocore
from datetime import datetime

from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.vision_retrain.vision_annotation_jobs import VisionAnnotationJob
from droidlet.tools.hitl.task_runner import TaskRunner


HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
VISION_LISTENER_POLL_TIME = 30

s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
logging.basicConfig(level="INFO")


class VisionListener(JobListener):
    """
    This data listener is responsible for spinning up Vision Annotation Jobs.

    Each Annotation Job is a batch of HITs where each turker is asked to mark which voxels correspond to a label.

    The steps are:
    1. Keep checking S3 for a vision error list associated with interaction HITs
    2. Create annotation jobs for those scenes
    3. Push annotation jobs to the runner, let runner schedule those annotation jobs

    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(VisionListener, self).__init__(timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
    
        while not self.check_is_finished():
            try:
                s3.Object(f"{S3_BUCKET_NAME}", f"{self._batch_id}/vision_errors.json").load()
            except botocore.exceptions.ClientError as e:
                logging.info(
                    f"[Interaction Vision Error Listener] No new data for {self._batch_id} ... Remaining time: {self.get_remaining_time()}"
                )
            else:
                logging.info(f"[Interaction Vision Error Listener] Vision error data found for {self._batch_id} ")
                response = s3.Object(f"{S3_BUCKET_NAME}", f"{self._batch_id}/vision_errors.json").get()
                scene_list = json.loads(response["Body"].read().decode("utf-8"))

                aj = VisionAnnotationJob(self._batch_id, timestamp=int(datetime.utcnow().timestamp()), scenes=scene_list, timeout=self.get_remaining_time())
                runner.register_data_generators([aj])

                logging.info(f"[Interaction Vision Error Listener] finished, shutting down listener [{self._batch_id}]")
                self.set_finished()

            if self.check_is_timeout():
                logging.info(f"[Interaction Vision Error Listener] timeout, shutting down {self._batch_id} listener")
                self.set_finished()
            time.sleep(VISION_LISTENER_POLL_TIME)


if __name__ == "__main__":
    runner = TaskRunner()
    vl = VisionListener(20220209160459, 30)
    runner.register_job_listeners([vl])
    runner.run()
