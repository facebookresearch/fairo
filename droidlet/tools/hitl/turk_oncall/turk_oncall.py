"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging

from oncall_jobs import OnCallJob
from command_lists import NUM_LISTS
from droidlet.tools.hitl.task_runner import TaskRunner

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oncall_job_num",
        type=int,
        default=1,
        help="number of oncall jobs to run (this number times the number of test command lists (17) is the number of HITs)",
    )
    parser.add_argument(
        "--image_tag",
        type=str,
        required=True,
        help="The tag of docker image that will be used to spin up ecs instance",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name of the ecs instance to be requested",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Number of minutes to wait before timeout",
    )
    opts = parser.parse_args()

    instance_num = opts.oncall_job_num * NUM_LISTS
    ocj = OnCallJob(instance_num, opts.image_tag, opts.task_name, timeout=opts.timeout)

    runner = TaskRunner()
    runner.register_data_generators([ocj])
    runner.run()
