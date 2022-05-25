"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import os
import signal
import subprocess
import time
import boto3
import yaml

from droidlet.tools.hitl.turk_oncall.allocate_oncall_instances import (
    allocate_oncall_instances,
    free_ecs_instances,
)
from droidlet.tools.hitl.utils.hitl_utils import (
    generate_batch_id,
    deregister_dashboard_subdomain,
)

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.task_runner import TaskRunner

from command_lists import COMMAND_LISTS

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"]
    if os.getenv("HITL_TMP_DIR")
    else f"{os.path.expanduser('~')}/.hitl"
)

S3_BUCKET_NAME = "droidlet-hitl"
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
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


class OnCallJob(DataGenerator):
    """
    This Data Generator is responsible for spinning up the Turk-As-Oncall version of Interaction Jobs.

    Each Interaction Job consists of several HITs. Turker are given dashboard sessions where they can
    send a set of predetermined commands to the craftassist agent and mark whether the response was as expected.

    On a high level:
    - The input of this data generator is a request specifying how many dashboard sessions are created for turkers
    - The output of this data generator is a set of dashboard logs with the results of each session.

    """

    def __init__(
        self, instance_num: int, image_tag: str, timeout: float = -1
    ) -> None:
        super(OnCallJob, self).__init__(timeout)
        self._instance_num = instance_num
        self._image_tag = image_tag
        self.instance_ids = None
        self.task_name = None
        self._batch_id = generate_batch_id()

    def run(self) -> None:
        batch_id = self._batch_id

        # Edit Mephisto config file task name
        with open(
            "../../crowdsourcing/turk_as_oncall/hydra_configs/conf/run_with_qual.yaml",
            "r",
        ) as stream:
            config = yaml.safe_load(stream)
            task_name = f"ca-oncall{self._batch_id}"
            self.task_name = task_name
            config["mephisto"]["task"]["task_name"] = task_name
        logging.info(
            f"Updating Mephisto config file to have task_name: {task_name}"
        )
        with open(
            "../../crowdsourcing/turk_as_oncall/hydra_configs/conf/run_with_qual.yaml",
            "w",
        ) as stream:
            stream.write("#@package _global_\n")
            yaml.dump(config, stream)

        # allocate AWS ECS instances and register DNS records
        logging.info(
            "Allocate AWS ECS instances, populate oncall data csv, and register DNS records..."
        )
        _, instance_ids = allocate_oncall_instances(
            self._instance_num,
            batch_id,
            self._image_tag,
            self.task_name,
            ECS_INSTANCE_TIMEOUT,
        )
        self.instance_ids = instance_ids

        # run Mephisto to spin up & monitor turk jobs
        logging.info("Start running Mephisto...")
        MEPHISTO_AWS_ACCESS_KEY_ID = os.environ["MEPHISTO_AWS_ACCESS_KEY_ID"]
        MEPHISTO_AWS_SECRET_ACCESS_KEY = os.environ[
            "MEPHISTO_AWS_SECRET_ACCESS_KEY"
        ]
        MEPHISTO_REQUESTER = os.environ["MEPHISTO_REQUESTER"]
        p = subprocess.Popen(
            [
                f"echo -ne '\n' |  \
                    AWS_ACCESS_KEY_ID='{MEPHISTO_AWS_ACCESS_KEY_ID}' \
                    AWS_SECRET_ACCESS_KEY='{MEPHISTO_AWS_SECRET_ACCESS_KEY}' \
                    python ../../crowdsourcing/turk_as_oncall/static_run_with_qual.py \
                    mephisto.provider.requester_name={MEPHISTO_REQUESTER} \
                    mephisto.architect.profile_name=mephisto-router-iam"
            ],
            shell=True,
            preexec_fn=os.setsid,
        )

        # Keep running Mephisto until timeout or job finished
        while not self.check_is_timeout() and p.poll() is None:
            logging.info(
                f"[Oncall Job] Oncall Job still running...Remaining time: {self.get_remaining_time()}"
            )
            time.sleep(INTERACTION_JOB_POLL_TIME)

        # if mephisto is still running after job timeout, terminate it
        logging.info("Manually terminate Mephisto after timeout...")
        if p.poll() is None:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(300)
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            time.sleep(300)
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        logging.info("Deregister DNS records...")
        deregister_dashboard_subdomain(batch_id)

        logging.info("Free ECS instances...")
        free_ecs_instances(self.instance_ids)

        logging.info(
            "Retrieving results from local Mephisto DB and uploading to S3"
        )
        self.get_local_db_results()

        self.set_finished()

    def get_batch_id(self):
        return self._batch_id

    def get_local_db_results(self):

        NUM_CMDS = 3
        results = [
            "worker_name|list_id|q1_result|q1_feedback|q2_result|q2_feedback|q3_result|q3_feedback|general_feedback\n"
        ]
        softblock = []
        command_dict = {}
        feedback_dict = {}
        summary_stats = []
        avg_duration = 0
        avg_usability = 0
        units = mephisto_data_browser.get_units_for_task_name(self.task_name)
        for unit in units:
            data = mephisto_data_browser.get_data_from_unit(unit)
            outputs = data["data"]["outputs"]

            # General stats
            worker_name = Worker.get(db, data["worker_id"]).worker_name
            avg_duration += (
                data["data"]["times"]["task_end"]
                - data["data"]["times"]["task_start"]
            ) / len(units)
            avg_usability += int(outputs["usability-rating"]) / len(units)

            # Task results
            results_string = f"{worker_name}|"
            list_id = int(outputs["listID"])
            results_string += f"{list_id}|"
            for i in range(NUM_CMDS):
                # Store the command result
                result = outputs[f"command_{i+1}"]
                results_string += f"{result}|"
                if command_dict.get((list_id, i), result) != result:
                    command_dict[(list_id, i)] = "disagree"
                else:
                    command_dict[(list_id, i)] = result

                # Store any command feedback
                cmd_feedback = outputs[f"command_{i+1}_feedback"]
                results_string += f"{cmd_feedback}|"
                if result == "no":
                    if not feedback_dict.get((list_id, i), None):
                        feedback_dict[(list_id, i)] = []
                    feedback_dict[(list_id, i)].append(cmd_feedback)

                    # If they said the agent failed but didn't tell us why, softblock
                    if not cmd_feedback:
                        softblock.append(f"{worker_name}\n")

            # Store general feedback
            results_string += f"{outputs['feedback']}"
            results.append(f"{results_string}\n")

        # Build the summary stats list
        summary_stats.append(f"Average Task Duration: {avg_duration:.2f}\n")
        summary_stats.append(
            f"Average Usability Rating: {avg_usability:.2f}\n"
        )
        summary_stats.append(
            f"Number of Commands Passed: {list(command_dict.values()).count('yes')}\n"
        )
        summary_stats.append(
            f"Number of Commands Failed: {list(command_dict.values()).count('no')}\n"
        )
        summary_stats.append(
            f"Number of Commands Disagreed: {list(command_dict.values()).count('disagree')}\n\n"
        )
        summary_stats.append(
            "List of failed commands (list_id|cmd_num|command_list|feedback):\n"
        )
        summary_stats.extend(
            [
                f"{key[0]}|{key[1]}|{COMMAND_LISTS[key[0]].replace('|', ',')}|{feedback_dict[key]}\n"
                for key in command_dict.keys()
                if command_dict[key] == "no"
            ]
        )

        # Log locally and upload results to S3
        local_dir = f"{HITL_TMP_DIR}/{self._batch_id}/oncall_results"
        os.makedirs(local_dir, exist_ok=True)

        with open(os.path.join(local_dir, "summary_stats.txt"), "w") as f:
            f.writelines(summary_stats)
        with open(os.path.join(local_dir, "results.txt"), "w") as f:
            f.writelines(results)
        with open(os.path.join(local_dir, "softblock.txt"), "w") as f:
            f.writelines(softblock)

        logging.info(f"Uploading job results to S3 folder: {self._batch_id}")
        s3.upload_file(
            f"{local_dir}/summary_stats.txt",
            S3_BUCKET_NAME,
            f"{self._batch_id}/summary_stats.txt",
        )
        s3.upload_file(
            f"{local_dir}/results.txt",
            S3_BUCKET_NAME,
            f"{self._batch_id}/results.txt",
        )
        s3.upload_file(
            f"{local_dir}/softblock.txt",
            S3_BUCKET_NAME,
            f"{self._batch_id}/softblock.txt",
        )


if __name__ == "__main__":
    runner = TaskRunner()
    ocj = OnCallJob(
        instance_num=2,
        image_tag="oncall_v1",
        task_name="oncall_t2",
        timeout=30,
    )
    batch_id = ocj.get_batch_id()
    runner.register_data_generators([ocj])
    runner.run()
