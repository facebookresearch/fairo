"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging

from retrain_nsp_jobs import InteractionJob, InteractionLogListener
from nsp_retrain_infra import NSPNewDataListener, NSPRetrainingJob
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
IJ_TIMEOUT = 30
IL_TIMEOUT = IJ_TIMEOUT + 20
NDL_TIMEOUT = IL_TIMEOUT + 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/scratch/ethancarlson/fairo/fairo/droidlet/")
    parser.add_argument("--full_data_dir", default="agents/craftassist/datasets/full_data/")
    parser.add_argument("--sweep_runner_dir", default="/checkpoint/ethancarlson/nsp_cl_scripts/")
    parser.add_argument("--sweep_config_folder", default="/checkpoint/ethancarlson/nsp/sweeps/scripts/configs/auto_sweep_configs/")
    parser.add_argument("--sweep_scripts_output_dir", default="/checkpoint/ethancarlson/nsp/sweeps/scripts/")
    parser.add_argument("--output_dir", default="/checkpoint/ethancarlson/nsp/sweeps/job_output")
    parser.add_argument("--checkpoint_dir", default="/checkpoint/ethancarlson/nsp/")
    parser.add_argument("--data_split_ratios", default="80/10/10", help="format - [train%]/[valid%]/[test%], set test to 0 to use only old data for testing")
    parser.add_argument("--new_data_training_threshold", default="100", help="number of new data samples below which no training occurs")
    parser.add_argument("--interaction_job_num", type=int, default=1, help="number of dashboard sessions to spin up")
    opts = parser.parse_args()
    # TODO Implement error handing are argument inputs

    # TODO: parameterize this
    instance_num = opts.interaction_job_num
    
    ij = InteractionJob(instance_num, timeout=IJ_TIMEOUT)
    batch_id = ij.get_batch_id()
    listener = InteractionLogListener(batch_id, IL_TIMEOUT)
    ndl = NSPNewDataListener(batch_id=batch_id, opts=opts, timeout=NDL_TIMEOUT)

    runner = TaskRunner()
    runner.register_data_generators([ij])
    runner.register_job_listeners([listener])
    runner.register_job_listeners([ndl])
    runner.run()