"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging

from interaction_jobs import InteractionJob, InteractionLogListener
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

# TODO: Parameterize those
# This specifies how long jobs should be running before we manually kill them
IJ_TIMEOUT = 360
IL_TIMEOUT = IJ_TIMEOUT + 20
NDL_TIMEOUT = IL_TIMEOUT + 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_job_num", type=int, default=2, help="number of dashboard sessions to spin up")
    parser.add_argument(
        "--image_tag", type=str, required=True, help="The tag of docker image that will be used to spin up ecs instance"
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Task name of the ecs instance to be requested"
    )
    opts = parser.parse_args()

    # TODO: parameterize this
    instance_num = opts.interaction_job_num
    image_tag = opts.image_tag
    task_name = opts.task_name
    
    ij = InteractionJob(instance_num, image_tag, task_name, timeout=IJ_TIMEOUT)
    #batch_id = ij.get_batch_id()
    #listener = InteractionLogListener(batch_id, IL_TIMEOUT)

    runner = TaskRunner()
    runner.register_data_generators([ij])
    #runner.register_job_listeners([listener])
    runner.run()