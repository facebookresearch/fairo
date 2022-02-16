"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import copy
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
from droidlet.tools.hitl.utils.allocate_instances import allocate_instances, free_ecs_instances
from droidlet.tools.hitl.utils.hitl_utils import generate_batch_id, deregister_dashboard_subdomain, dedup_commands
from droidlet.tools.hitl.utils.process_s3_logs import read_s3_bucket, read_turk_logs

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner

from droidlet.tools.crowdsourcing.droidlet_static_html_task.issue_bonus import issue_bonuses

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

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
    """
    This Data Generator is responsible for spinning up Interaction Jobs.

    Each Interaction Job consists of several HITs. Turker are given dashboard sessions where they can
    interact with craftassist agent for a period of time and annotate error commands they observed.

    On a high level:
    - The input of this data generator is a request specifying how many dashboard sessions are created for turkers
    - The output of this data generator is a list of (error) commands collected from all dashboard sessions

    """

    def __init__(self, instance_num: int, image_tag: str, task_name: str, timeout: float = -1) -> None:
        super(InteractionJob, self).__init__(timeout)
        self._instance_num = instance_num
        self._image_tag = image_tag
        self._task_name = task_name
        self.instance_ids = None
        self._batch_id = generate_batch_id()

    def run(self) -> None:
        batch_id = self._batch_id

        # allocate AWS ECS instances and register DNS records
        logging.info(f"Allocate AWS ECS instances and register DNS records...")
        _, instance_ids = allocate_instances(self._instance_num, batch_id, self._image_tag, self._task_name, ECS_INSTANCE_TIMEOUT)
        self.instance_ids = instance_ids

        # run Mephisto to spin up & monitor turk jobs
        logging.info(f"Start running Mephisto...")
        MEPHISTO_AWS_ACCESS_KEY_ID = os.environ["MEPHISTO_AWS_ACCESS_KEY_ID"]
        MEPHISTO_AWS_SECRET_ACCESS_KEY = os.environ["MEPHISTO_AWS_SECRET_ACCESS_KEY"]
        MEPHISTO_REQUESTER = os.environ["MEPHISTO_REQUESTER"]
        p = subprocess.Popen(
            [
                f"echo -ne '\n' |  AWS_ACCESS_KEY_ID='{MEPHISTO_AWS_ACCESS_KEY_ID}' AWS_SECRET_ACCESS_KEY='{MEPHISTO_AWS_SECRET_ACCESS_KEY}' python ../../crowdsourcing/droidlet_static_html_task/static_run_with_qual.py mephisto.provider.requester_name={MEPHISTO_REQUESTER}"
            ],
            shell=True,
            preexec_fn=os.setsid,
        )

        # Keep running Mephisto until timeout or job finished
        while not self.check_is_timeout() and p.poll() is None:
            logging.debug(
                f"[Interaction Job] Interaction Job still running...Remaining time: {self.get_remaining_time()}"
            )
            time.sleep(INTERACTION_JOB_POLL_TIME)

        # if mephisto is still running after job timeout, terminate it
        logging.info(f"Manually terminate Mephisto after timeout...")
        if p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(300)
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(300)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        # Deregister DNS records
        logging.info(f"Deregister DNS records...")
        deregister_dashboard_subdomain(batch_id)

        logging.info(f"Free ECS instances...")
        free_ecs_instances(self.instance_ids)

        logging.info(f"Issuing performance incentive bonuses...")
        issue_bonuses(self._task_name)

        logging.info(f"Processing S3 logs...")
        self.process_s3_logs(batch_id)

        self.set_finished()

    def process_s3_logs(self, batch_id) -> None:
        """
        This should be called after all dashboard sessions have completed.
        It will read session data from S3, parse log files, collating commands into a single list
        and finally upload the list to S3.
        """
        s3_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/turk_logs")
        parsed_logs_dir = os.path.join(HITL_TMP_DIR, f"{batch_id}/parsed_turk_logs")
        os.makedirs(s3_logs_dir, exist_ok=True)
        os.makedirs(parsed_logs_dir, exist_ok=True)

        time.sleep(120)
        rc = subprocess.call(
            [f"aws s3 sync {S3_ROOT}/{batch_id}/interaction {s3_logs_dir}"], shell=True
        )
        time.sleep(120)

        read_s3_bucket(s3_logs_dir, parsed_logs_dir)
        command_list = read_turk_logs(parsed_logs_dir, NSP_OUTPUT_FNAME)
        logging.info(f"command list from interactions: {command_list}")

        logging.info(f"Uploading command list to S3...")
        content = "\n".join(command_list)
        s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/collected_commands").put(Body=content)
        content = "commands_ready"
        s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/commands_ready").put(Body=content)

    def get_batch_id(self):
        return self._batch_id


class InteractionLogListener(JobListener):
    """
    This Listener is responsible for listening to new commands and create annotation jobs.

    The steps are:
    1. Keep checking S3 for any new, unannotated commands
    2. Create annotation jobs for those unannotated commands
    3. Push annotation jobs to the runner, let runner schedule those annotation jobs
    4. Before it shuts down, collating all annotated commands and appending them to the central list,
       also keep track of the indexes of the commands in the central list in a meta.txt file. This file
       can be used together with central list to create training data & masks for NSP retraining job

    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(InteractionLogListener, self).__init__(timeout=timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
        batch_id = self._batch_id

        # Keep checking if there is any new command uploaded to S3. Once found,
        # create an annotation job for the command and register the job to the runner
        while not self.check_is_finished():
            try:
                s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/commands_ready").load()
                s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/collected_commands").load()
            except botocore.exceptions.ClientError as e:
                logging.info(
                    f"[Interaction Log Listener] No new data for {batch_id} ... Remaining time: {self.get_remaining_time()}"
                )
            else:
                response = s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/collected_commands").get()
                commands = response["Body"].read().decode("utf-8").split("\n")
                cmd_id = 0
                cmd_list = dedup_commands(commands)
                for cmd in cmd_list:
                    logging.info(
                        f"Pushing Annotation Job [{batch_id}-{cmd_id}-{cmd}] to runner..."
                    )
                    annotation_job = AnnotationJob(
                        batch_id, cmd, cmd_id, self.get_remaining_time()
                    )
                    runner.register_data_generators([annotation_job])
                    cmd_id += 1

                s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/commands_ready").delete()

            if self.check_is_timeout():
                self.set_finished()
            time.sleep(INTERACTION_LISTENER_POLL_TIME)

        logging.info(f"[Interaction Log Listener] Finished, collate and upload data to S3")
        # Finally, consolidate all annotated data and upload to s3
        # First, get all annotated commands of this run
        annotated_pairs = []
        bucket = s3.Bucket(S3_BUCKET_NAME)
        for obj in bucket.objects.filter(Prefix=f"{batch_id}/"):
            try:
                fname = obj.key
                if "_all_combined.txt" in fname:
                    response = s3.Object(f"{S3_BUCKET_NAME}", f"{fname}").get()
                    annotated_pairs.append(
                        response["Body"].read().decode("utf-8").split("\n")[0].split("\t")
                    )
            except:
                logging.info(f"Something went wrong when retrieving {obj.key}")

        # Then, get all annotated commands from all runs (including previous runs)
        # Doing this because we are appending annotated commands to a central list
        # and keeping track of the command indexes of this run in the central list
        prev_annotated_pairs_with_idx = []
        try:
            response = s3.Object(f"{S3_BUCKET_NAME}", f"{ANNOTATED_COMMANDS_FNAME}").get()
            lines = response["Body"].read().decode("utf-8").split("\n")
            prev_annotated_pairs_with_idx = [line.split("|") for line in lines]
        except:
            logging.info(f"No previous annotated commands found...")

        # Appending commands of this run to the central list
        idx = len(prev_annotated_pairs_with_idx)
        idx_list = []
        annotated_pairs_with_idx = copy.deepcopy(prev_annotated_pairs_with_idx)
        for annotated_pair in annotated_pairs:
            annotated_pairs_with_idx.append([str(idx), annotated_pair[0], annotated_pair[1]])
            idx_list.append(str(idx))
            idx += 1

        # Finally, upload updated data and meta file to S3
        lines = ["|".join(triple) for triple in annotated_pairs_with_idx]
        data_content = "\n".join(lines)
        meta_content = "\n".join(idx_list)
        try:
            s3.Object(f"{S3_BUCKET_NAME}", f"{ANNOTATED_COMMANDS_FNAME}").put(Body=data_content)
            s3.Object(f"{S3_BUCKET_NAME}", f"{batch_id}/meta.txt").put(Body=meta_content)
        except:
            logging.info("[Interaction Log Listener] Err on uploading annotated data to S3...")


if __name__ == "__main__":
    runner = TaskRunner()
    ij = InteractionJob(1, timeout=10)
    batch_id = ij.get_batch_id()
    listener = InteractionLogListener(batch_id, 15)
    runner.register_data_generators([ij])
    runner.register_job_listeners([listener])
    runner.run()
