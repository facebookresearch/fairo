"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import os
import signal
import subprocess
import time
import csv
import yaml

import boto3

from droidlet.tools.hitl.data_generator import DataGenerator
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
ANNOTATION_JOB_POLL_TIME = 30
ANNOTATION_PROCESS_TIMEOUT_DEFAULT = 7200
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
MEPHISTO_REQUESTER = os.environ["MEPHISTO_REQUESTER"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

logging.basicConfig(level="INFO")


class VisionAnnotationJob(DataGenerator):
    """
    This Data Generator is responsible for spinning up Vision Annotation Jobs.

    Each Vision Annotation Job is a batch of HITs where turkers are asked to annotate the visual scene
    captured at the time a command was issued and labeled as containing a perception error.

    On a high level:
    - The inputs of this data generator are a list of scenes and corresponding object labels
    - The output of this data generator is the same visual scene and label, with a new field representing
      the instance segmentation mask.

    """

    def __init__(self, batch_id: int, scene_list: list, label_list: list, timeout: float = -1) -> None:
        super(VisionAnnotationJob, self).__init__(timeout)
        self._batch_id = batch_id
        self._scene_list = scene_list
        self._label_list = label_list

    def run(self) -> None:
        try:
            # Retrieve the scenes from S3 and put them in extra_refs
            download_key = str(self._batch_id) + "/vision_annotation/scene_list.json"
            scene_ref_filepath = os.path.join(os.getcwd(), "../../crowdsourcing/vision_annotation_task/server_files/extra_refs/scene_list.json")
            s3.download_file('droidlet-hitl', download_key, scene_ref_filepath)
            logging.info(f"List of batch scenes downloaded from S3")

            # Write scene indeces and labels to data.csv for Mephisto to read
            with open("../../crowdsourcing/vision_annotation_task/data.csv", "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow(["batch_id", "scene_idx", "label"])
                for i in range(len(self._scene_list)):
                    csv_writer.writerow([str(self._batch_id), str(self._scene_list[i]), self._label_list[i]])

            # Edit Mephisto config file task name
            with open("../../crowdsourcing/vision_annotation_task/conf/annotation.yaml", "r") as stream:
                config = yaml.safe_load(stream)
                task_name = "ca-vis-anno" + str(self._batch_id)
                config["mephisto"]["task"]["task_name"] = task_name
            logging.info(f"Updating Mephisto config file to have task_name: {task_name}")
            with open("../../crowdsourcing/vision_annotation_task/conf/annotation.yaml", "w") as stream:
                stream.write("#@package _global_\n")
                yaml.dump(config, stream)

            # Launch the batch of HITs
            anno_job_path = os.path.join(os.getcwd(), "../../crowdsourcing/vision_annotation_task/run_annotation_with_qual.py")
            anno_cmd = "python3 " + anno_job_path + \
                " mephisto.provider.requester_name=" + MEPHISTO_REQUESTER + \
                " mephisto.architect.profile_name=mephisto-router-iam"
            p = subprocess.Popen(anno_cmd, shell=True, preexec_fn=os.setsid)

            # Keep running Mephisto until timeout or job finished
            while not self.check_is_timeout() and p.poll() is None:
                logging.info(f"Vision Annotation Job [{self._batch_id}] still running...Remaining time: {self.get_remaining_time()}")
                time.sleep(ANNOTATION_JOB_POLL_TIME)

            if p.poll() is None:
                # If mturk job is still running after timeout, terminate it
                logging.info(f"Manually terminate turk job after timeout...")
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
                time.sleep(10)
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            
            # Retrieve annotated scene data from Mephisto DB
            os.makedirs(os.path.join(HITL_TMP_DIR, str(self._batch_id)), exist_ok=True)
            results_csv = os.path.join(HITL_TMP_DIR, f"{self._batch_id}/annotated_scenes.csv")
            logging.info(f"Retrieving data from Mephisto DB and writing to {results_csv}")
            with open(results_csv, "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow(["batch_id", "scene_idx", "label", "inst_seg_tags"])

                units = mephisto_data_browser.get_units_for_task_name(task_name)
                for unit in units:
                    data = mephisto_data_browser.get_data_from_unit(unit)
                    scene_idx = data["data"]["outputs"]["scene_idx"]
                    label = data["data"]["outputs"]["label"]
                    inst_seg_tags = data["data"]["outputs"]["inst_seg_tags"]
                    csv_writer.writerow([str(self._batch_id), scene_idx, label, inst_seg_tags])

            # Upload vision annotation results to S3
            logging.info(f"Uploading scene annotation data to S3: {self._batch_id}/vision_annotation/annotated_scenes.csv")
            with open(results_csv, "rb") as f:
                s3.upload_fileobj(f, f"{S3_BUCKET_NAME}", f"{self._batch_id}/vision_annotation/annotated_scenes.csv")
            logging.info(f"Uploading completed")

            # Delete the scene file from extra_refs
            os.remove(scene_ref_filepath)

        except:
            logging.info(f"Annotation Job [{self._batch_id}] terminated unexpectedly...")

        self.set_finished()


if __name__ == "__main__":
    aj = VisionAnnotationJob(20220119175234, [0], ["grey floating cube"], 300)
    aj.run()
