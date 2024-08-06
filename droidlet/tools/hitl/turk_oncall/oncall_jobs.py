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
import glob
import json
import urllib.parse

from droidlet.tools.hitl.turk_oncall.allocate_oncall_instances import (
    allocate_oncall_instances,
    free_ecs_instances,
)
from droidlet.tools.hitl.utils.hitl_utils import (
    generate_batch_id,
    deregister_dashboard_subdomain,
)
from droidlet.tools.hitl.utils.process_s3_logs import read_s3_bucket

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.hitl.turk_oncall.oncall_bug_report import TaoLogListener

from command_lists import COMMAND_LISTS

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.data_model.worker import Worker

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

S3_BUCKET_NAME = "droidlet-hitl"
S3_BASE_URL = (
    f"https://s3.console.aws.amazon.com/s3/object/{S3_BUCKET_NAME}?region=us-west-2&prefix="
)
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

    def __init__(self, instance_num: int, image_tag: str, timeout: float = -1) -> None:
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
        logging.info(f"Updating Mephisto config file to have task_name: {task_name}")
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
        MEPHISTO_AWS_SECRET_ACCESS_KEY = os.environ["MEPHISTO_AWS_SECRET_ACCESS_KEY"]
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

        logging.info("Extracting and processing S3 logs")
        agent_logs_map = self.process_s3_logs()

        logging.info("Retrieving local Mephisto DB results and uploading summary")
        self.get_local_db_results(agent_logs_map)

        # Add stat file
        stat_fname = f"{batch_id}.stat"

        obj = s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}")
        result = obj.put(Body="ready").get("ResponseMetadata")
        if result.get("HTTPStatusCode") != 200:
            logging.info(f"[Oncall Job] {batch_id}.stat not updated")
        else:
            # Add a listener
            tao_log_listener = TaoLogListener(batch_id)
            tao_log_listener.add_parent_jobs([self])
            runner.register_job_listeners([tao_log_listener])
            self.set_finished()

    def get_batch_id(self):
        return self._batch_id

    def get_local_db_results(self, agent_logs_map):
        NUM_CMDS = 3
        results = [
            "worker_name|logs_directory|list_id|q1_result|q1_feedback|q2_result|q2_feedback|q3_result|q3_feedback|general_feedback\n"
        ]
        softblock = []
        summary_stats = []
        command_lut = {"result": {}, "feedback": {}, "logs": {}}
        avg_duration = avg_usability = 0
        units = mephisto_data_browser.get_units_for_task_name(self.task_name)
        for unit in units:
            data = mephisto_data_browser.get_data_from_unit(unit)
            outputs = data["data"]["outputs"]

            # General stats
            worker_name = Worker.get(db, data["worker_id"]).worker_name
            avg_duration += (
                data["data"]["times"]["task_end"] - data["data"]["times"]["task_start"]
            ) / len(units)
            avg_usability += int(outputs["usability-rating"]) / len(units)

            # Task results
            results_string = f"{worker_name}|"
            agent = unit.get_assigned_agent().get_agent_id()
            results_string += f"{agent_logs_map[agent]}|"
            list_id = int(outputs["listID"])
            results_string += f"{list_id}|"
            for i in range(NUM_CMDS):
                # Store the command result
                result = outputs[f"command_{i+1}"]
                results_string += f"{result}|"
                if command_lut["result"].get((list_id, i), result) != result:
                    command_lut["result"][(list_id, i)] = "disagree"
                else:
                    command_lut["result"][(list_id, i)] = result

                # Store any command feedback
                cmd_feedback = outputs[f"command_{i+1}_feedback"]
                results_string += f"{cmd_feedback}|"
                if result == "no":
                    if not command_lut["feedback"].get((list_id, i), None):
                        command_lut["feedback"][(list_id, i)] = []
                    command_lut["feedback"][(list_id, i)].append(cmd_feedback)

                    # Store logs URLs for failed commands
                    if not command_lut["logs"].get((list_id, i), None):
                        command_lut["logs"][(list_id, i)] = []
                    log_url = f"{S3_BASE_URL}{batch_id}/interaction/{urllib.parse.quote_plus(agent_logs_map[agent])}/logs.tar.gz"
                    command_lut["logs"][(list_id, i)].append(log_url)

                    # If they said the agent failed but didn't tell us why, softblock
                    if not cmd_feedback:
                        softblock.append(f"{worker_name}\n")

            # Store general feedback
            results_string += f"{outputs['feedback']}"
            results.append(f"{results_string}\n")

        # Build the summary stats list
        summary_stats.append(f"Average Task Duration: {avg_duration:.2f}\n")
        summary_stats.append(f"Average Usability Rating: {avg_usability:.2f}\n")
        summary_stats.append(
            f"Number of Commands Passed: {list(command_lut['result'].values()).count('yes')}\n"
        )
        summary_stats.append(
            f"Number of Commands Failed: {list(command_lut['result'].values()).count('no')}\n"
        )
        summary_stats.append(
            f"Number of Commands Disagreed: {list(command_lut['result'].values()).count('disagree')}\n\n"
        )

        summary_stats.append(
            "List of failed commands (list_id|cmd_num|command_list|feedback|logs_url):\n"
        )
        summary_stats.extend(
            [
                f"{key[0]}|{key[1]}|{COMMAND_LISTS[key[0]].replace('|', ',')}|{command_lut['feedback'][key]}|{command_lut['logs'][key]}\n"
                for key in command_lut["result"].keys()
                if command_lut["result"][key] == "no"
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

    def process_s3_logs(self) -> dict:
        """
        This reads the local logs directory after all tarfiles have been extracted.
        It returns a map of Mephisto Agent IDs to the directory names of the logs they produced.
        """
        s3_logs_dir = os.path.join(HITL_TMP_DIR, f"{self._batch_id}/turk_logs")
        parsed_logs_dir = os.path.join(HITL_TMP_DIR, f"{self._batch_id}/parsed_turk_logs")
        os.makedirs(s3_logs_dir, exist_ok=True)
        os.makedirs(parsed_logs_dir, exist_ok=True)

        subprocess.call(
            [f"aws s3 sync {S3_ROOT}/{self._batch_id}/interaction {s3_logs_dir}"],
            shell=True,
        )
        logging.info("Waiting 2min for S3 directory sync to finish")
        time.sleep(120)

        # Extract tarfiles
        read_s3_bucket(s3_logs_dir, parsed_logs_dir)

        # Build a map of agent ids to logs directories
        agent_logs_map = {}
        for json_path in glob.glob(f"{parsed_logs_dir}/**/job_metadata.json"):
            with open(json_path, "r") as f:
                js = json.load(f)

            dir = os.path.basename(os.path.dirname(json_path))
            agent_logs_map[js["mephisto_agent_id"]] = dir

        return agent_logs_map


if __name__ == "__main__":
    runner = TaskRunner()
    ocj = OnCallJob(
        instance_num=2,
        image_tag="oncall_v1",
        timeout=30,
    )
    batch_id = ocj.get_batch_id()
    runner.register_data_generators([ocj])
    runner.run()
