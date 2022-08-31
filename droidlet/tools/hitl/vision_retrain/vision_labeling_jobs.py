"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import subprocess
import boto3
from datetime import datetime
import csv
import sys
import yaml
import logging
import json
import time
import signal
import argparse

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.hitl.utils.hitl_utils import generate_batch_id
from droidlet.tools.hitl.vision_retrain.vision_annotation_jobs import VisionAnnotationJob

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker


db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)
s3 = boto3.client("s3")
SCENE_GEN_TIMEOUT = 60
SCENE_GEN_POLL_TIME = 10
VISION_LABELING_POLL_TIME = 30
VIS_LABELING_LISTENER_POLL_TIME = 15

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
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


class VisionLabelingJob(DataGenerator):
    """
    This Data Generator is responsible for spinning up voxel world object labeling jobs.

    Each Labeling Job is a HIT where turkers are asked to name an object that they see in the scene.

    On a high level:
    - The input of this data generator is a request specifying how many scenes to label (HITs to launch)
    - The output of this data generator is a list of scene dicts containing object labels in the `obj_ref` field

    """

    def __init__(self, opts, timeout: float = -1) -> None:
        super(VisionLabelingJob, self).__init__(timeout)
        self._batch_id = generate_batch_id()
        self._SL = opts.scene_length
        self._H = opts.scene_height
        self._GROUND_DEPTH = opts.ground_depth
        self._MAX_NUM_SHAPES = opts.max_num_shapes
        self._NUM_HITS = opts.num_hits
        self._NUM_SCENES = int(opts.num_hits / opts.hits_per_scene)
        self._MAX_HOLES = opts.max_num_holes
        self._use_basic_shapes = opts.use_basic_shapes
        self._num_iglu_scenes = opts.num_iglu_scenes

    def run(self) -> None:

        os.makedirs(f"{HITL_TMP_DIR}/{self._batch_id}/vision_labeling", exist_ok=True)

        try:
            # Generate scenes
            scene_save_path = os.path.join(
                os.getcwd(),
                "../../crowdsourcing/vision_annotation_task/server_files/extra_refs/scene_list.json",
            )
            scene_gen_path = os.path.join(
                os.getcwd(), "../../../lowlevel/minecraft/small_scenes_with_shapes.py"
            )
            scene_gen_cmd = (
                "python3 "
                + scene_gen_path
                + " --SL="
                + str(self._SL)
                + " --H="
                + str(self._H)
                + " --GROUND_DEPTH="
                + str(self._GROUND_DEPTH)
                + " --MAX_NUM_SHAPES="
                + str(self._MAX_NUM_SHAPES)
                + " --NUM_SCENES="
                + str(self._NUM_SCENES)
                + " --MAX_NUM_GROUND_HOLES="
                + str(self._MAX_HOLES)
                + " --save_data_path="
                + scene_save_path
            )
            if not self._use_basic_shapes:
                scene_gen_cmd += " --iglu_scenes=" + os.environ["IGLU_SCENE_PATH"]
                scene_gen_cmd += " --num_iglu_scenes=" + str(self._num_iglu_scenes)
            try:
                logging.info("Starting scene generation script")
                scene_gen = subprocess.Popen(
                    scene_gen_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True
                )
            except ValueError:
                logging.info("Likely error: Popen called with invalid arguments")
                raise

            # Keep running Mephisto until timeout or job finished
            now = datetime.now().timestamp()
            while (
                not (datetime.now().timestamp() - now > SCENE_GEN_TIMEOUT)
                and scene_gen.poll() is None
            ):
                logging.info(
                    f"Scene generation script still running...Remaining time: {int(SCENE_GEN_TIMEOUT - (datetime.now().timestamp() - now))} seconds"
                )
                time.sleep(SCENE_GEN_POLL_TIME)

            if scene_gen.poll() is None:
                # if scene generator is still running after timeout, terminate it
                logging.info("Scene generation timed out, canceling labeling job!")
                os.killpg(os.getpgid(scene_gen.pid), signal.SIGINT)
                time.sleep(10)
                os.killpg(os.getpgid(scene_gen.pid), signal.SIGKILL)
                self.set_finished()

            # Upload original scene list to S3 for later comparison
            upload_key = f"{self._batch_id}/vision_labeling_results/{self._batch_id}original_scene_list.json"
            logging.info(f"Uploading generated scene list to S3: {upload_key}")
            response = s3.upload_file(scene_save_path, "droidlet-hitl", upload_key)
            if response:
                logging.info("S3 response: " + response)

            # Populate labeling_data.csv with scene filename and indeces for internal Mephisto per-HIT reference
            logging.info(
                "Populating 'labeling_data.csv' with scene references for internal Mephisto use"
            )
            with open("../../crowdsourcing/vision_annotation_task/labeling_data.csv", "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow(["scene_filename", "scene_idx"])
                for i in range(self._NUM_HITS):
                    csv_writer.writerow(["scene_list.json", str(i % self._NUM_SCENES)])

            # Edit Mephisto config file task name
            logging.info(
                "Editing Mephisto config file to have parameterized task anme and units/worker"
            )
            maximum_units_per_worker = 10 if self._NUM_HITS < 50 else int(self._NUM_HITS / 8)
            with open(
                "../../crowdsourcing/vision_annotation_task/conf/labeling.yaml", "r"
            ) as stream:
                config = yaml.safe_load(stream)
                task_name = f"ca-vis-label{self._batch_id}"
                config["mephisto"]["task"]["task_name"] = task_name
                config["mephisto"]["task"]["maximum_units_per_worker"] = maximum_units_per_worker
            logging.info(f"Updating Mephisto config file to have task_name: {task_name}")
            with open(
                "../../crowdsourcing/vision_annotation_task/conf/labeling.yaml", "w"
            ) as stream:
                stream.write("#@package _global_\n")
                yaml.dump(config, stream)

            # Launch via Mephisto
            MEPHISTO_REQUESTER = os.environ["MEPHISTO_REQUESTER"]
            job_launch_cmd = (
                "echo -ne '\n' | python3 ../../crowdsourcing/vision_annotation_task/run_labeling_with_qual.py"
                + " mephisto.provider.requester_name="
                + MEPHISTO_REQUESTER
                + " mephisto.architect.profile_name=mephisto-router-iam"
            )
            try:
                logging.info(f"Launching job with {self._NUM_HITS} HITs")
                job_launch = subprocess.Popen(job_launch_cmd, shell=True, preexec_fn=os.setsid)
            except ValueError:
                logging.info("Likely error: Popen called with invalid arguments")
                raise

            # Keep running Mephisto until timeout or job finished
            while not self.check_is_timeout() and job_launch.poll() is None:
                logging.info(
                    f"Vision Labeling Job [{self._batch_id}] still running...Remaining time: {self.get_remaining_time()}"
                )
                time.sleep(VISION_LABELING_POLL_TIME)

            if job_launch.poll() is None:
                # if mturk job is still running after timeout, terminate it
                logging.info("Manually terminate turk job after timeout...")
                os.killpg(os.getpgid(job_launch.pid), signal.SIGINT)
                time.sleep(180)
                os.killpg(os.getpgid(job_launch.pid), signal.SIGINT)
                time.sleep(120)
                os.killpg(os.getpgid(job_launch.pid), signal.SIGKILL)

            # Pull results from local DB
            logging.info(
                f"Mephisto labeling job [{self._batch_id}] complete, retreiving results from local DB"
            )
            results_csv = f"{HITL_TMP_DIR}/{self._batch_id}/vision_labeling/{self._batch_id}.csv"
            with open(results_csv, "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerow(
                    ["scene_filename", "scene_idx", "worker_name", "object", "location"]
                )

                units = mephisto_data_browser.get_units_for_task_name(task_name)
                scene_list = []
                for unit in units:
                    data = mephisto_data_browser.get_data_from_unit(unit)
                    worker_name = Worker.get(db, data["worker_id"]).worker_name
                    outputs = data["data"]["outputs"]
                    csv_writer.writerow(
                        [
                            outputs["scene_filename"],
                            outputs["scene_idx"],
                            worker_name,
                            outputs["object"],
                            outputs["location"],
                        ]
                    )

                    # Build the list of scenes and populate the obj_ref (label) field
                    with open(
                        f"../../crowdsourcing/vision_annotation_task/server_files/extra_refs/{outputs['scene_filename']}",
                        "r",
                    ) as js:
                        scene = json.load(js)[int(outputs["scene_idx"])]
                        scene["obj_ref"] = outputs["object"]
                        scene_list.append(scene)

            # Upload results to S3
            upload_key = f"{self._batch_id}/vision_labeling_results/{results_csv}"
            logging.info(f"Uploading job results to S3: {upload_key}")
            response = s3.upload_file(results_csv, S3_BUCKET_NAME, upload_key)
            if response:
                logging.info("S3 response: " + response)

            # Upload scene file to S3 (this is what data listener looks for)
            upload_key = f"{self._batch_id}/vision_labeling_results/scene_list.json"
            logging.info(f"Uploading job results to S3: {upload_key}")
            labeled_scene_list = (
                f"{HITL_TMP_DIR}/{self._batch_id}/vision_labeling/labeled_scene_list.json"
            )
            with open(labeled_scene_list, "w") as f:
                json.dump(scene_list, f)
            with open(labeled_scene_list, "rb") as f:
                response = s3.upload_fileobj(f, S3_BUCKET_NAME, upload_key)
                if response:
                    logging.info("S3 response: " + response)

            logging.info(f"Labeling job {self._batch_id} complete")

        except:
            logging.info(f"Vision Labeling Job [{self._batch_id}] terminated unexpectedly...")
            raise

        self.set_finished()

    def get_batch_id(self):
        return self._batch_id


class VisionLabelingListener(JobListener):
    """
    This Listener is responsible for listening for a batch of labeled voxel scenes from the above data generator

    The steps are:
    1. Keep checking S3 for a scene list associated with vision labeling jobs
    2. Create annotation jobs for those labeled scenes
    3. Push annotation jobs to the runner, let runner schedule those annotation jobs

    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(VisionLabelingListener, self).__init__(timeout=timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
        batch_id = self._batch_id

        # Keep checking if there is any new scene list uploaded to S3 associated with batch_id.
        # Once found, create an annotation job for the command and register the job to the runner
        while not self.check_is_finished():
            try:
                with open("scene_list.json", "wb") as js:
                    s3.download_fileobj(
                        f"{S3_BUCKET_NAME}",
                        f"{batch_id}/vision_labeling_results/scene_list.json",
                        js,
                    )
            except:
                logging.info(
                    f"[Vision Labeling Job Listener] No new data for {batch_id} ... Remaining time: {self.get_remaining_time()}"
                )
            else:
                logging.info(
                    f"[Vision Labeling Job Listener] data found for batch_id [{batch_id}], downloading new labeled data"
                )
                with open("scene_list.json", "r") as js:
                    scene_list = json.load(js)
                logging.info(
                    f"[Vision Labeling Job Listener] Data downloaded, pushing {batch_id} annotation job to runner..."
                )

                os.remove("scene_list.json")

                aj = VisionAnnotationJob(
                    batch_id=batch_id,
                    timestamp=int(datetime.utcnow().timestamp()),
                    scenes=scene_list,
                    timeout=self.get_remaining_time(),
                )
                runner.register_data_generators([aj])

                logging.info(
                    f"[Vision Labeling Job Listener] finished, shutting down listener [{batch_id}]"
                )
                self.set_finished()

            if self.check_is_timeout():
                logging.info(
                    f"[Vision Labeling Job Listener] timeout, shutting down {batch_id} listener"
                )
                self.set_finished()
            time.sleep(VIS_LABELING_LISTENER_POLL_TIME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_length", type=int, default=17)
    parser.add_argument("--scene_height", type=int, default=13)
    parser.add_argument("--ground_depth", type=int, default=4)
    parser.add_argument("--max_num_shapes", type=int, default=4)
    parser.add_argument("--max_num_holes", type=int, default=3)
    parser.add_argument("--num_hits", type=int, default=100, help="Number of HITs to request")
    parser.add_argument(
        "--labeling_timeout",
        type=int,
        default=75,
        help="Number of minutes before labeling job times out",
    )
    parser.add_argument(
        "--annotation_timeout",
        type=int,
        default=75,
        help="Number of minutes before annotation job times out",
    )
    parser.add_argument("--use_basic_shapes", action="store_true", default=False)
    parser.add_argument("--num_iglu_scenes", type=int, default=30, help="Subset of IGLU scenes")
    parser.add_argument(
        "--hits_per_scene", type=int, default=10, help="# HITs to launch per scene"
    )
    opts = parser.parse_args()

    LISTENER_TIMEOUT = opts.labeling_timeout + opts.annotation_timeout

    runner = TaskRunner()
    lj = VisionLabelingJob(opts, opts.labeling_timeout)

    batch_id = lj.get_batch_id()
    listener = VisionLabelingListener(batch_id, LISTENER_TIMEOUT)
    runner.register_data_generators([lj])
    runner.register_job_listeners([listener])
    runner.run()
