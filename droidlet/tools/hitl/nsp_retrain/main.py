"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging

from interaction_jobs import InteractionJob, InteractionLogListener
from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.hitl.hitl_logger import set_logging_root_path

HITL_LOG_DIR = ""
set_logging_root_path(HITL_LOG_DIR)

# TODO: Parameterize those
# This specifies how long jobs should be running before we manually kill them
IJ_TIMEOUT = 10
IL_TIMEOUT = IJ_TIMEOUT + 10
NDL_TIMEOUT = IL_TIMEOUT + 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_job_num", type=int, default=2, help="number of dashboard sessions to spin up")
    opts = parser.parse_args()

    # TODO: parameterize this
    instance_num = opts.interaction_job_num
    
    ij = InteractionJob(instance_num, timeout=IJ_TIMEOUT)
    batch_id = ij.get_batch_id()
    listener = InteractionLogListener(batch_id, IL_TIMEOUT)

    runner = TaskRunner()
    runner.register_data_generators([ij])
    runner.register_job_listeners([listener])
    runner.run()