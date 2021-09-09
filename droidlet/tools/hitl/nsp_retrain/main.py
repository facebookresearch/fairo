"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging

from retrain_nsp_jobs import InteractionJob, InteractionLogListener
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
        "--instance_num", type=int, default=1, help="number of instances requested"
    )
    args = parser.parse_args()

    interaction_job = InteractionJob(args.instance_num, 0.01)
    batch_id = interaction_job.get_batch_id()
    interaction_listener = InteractionLogListener(batch_id)
    runner = TaskRunner()
    runner.register_data_generators([interaction_job])
    runner.register_job_listeners([interaction_listener])
    runner.run()
