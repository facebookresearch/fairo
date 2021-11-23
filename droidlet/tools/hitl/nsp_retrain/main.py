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
IJ_TIMEOUT = 720
IL_TIMEOUT = IJ_TIMEOUT + 360
NDL_TIMEOUT = IL_TIMEOUT + 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/private/home/yuxuans/Workspace/hitl_baseline/fairo/")
    parser.add_argument("--full_data_dir", default="agents/craftassist/datasets/full_data/")
    parser.add_argument("--sweep_runner_dir", default="/checkpoint/yuxuans/nsp_cl_scripts/")
    parser.add_argument("--sweep_config_folder", default="/checkpoint/yuxuans/nsp/sweeps/scripts/configs/auto_sweep_configs/")
    parser.add_argument("--sweep_scripts_output_dir", default="/checkpoint/yuxuans/nsp/sweeps/scripts/")
    parser.add_argument("--output_dir", default="/checkpoint/yuxuans/nsp/sweeps/job_output/")
    parser.add_argument("--checkpoint_dir", default="/checkpoint/yuxuans/nsp/")
    parser.add_argument("--data_split_ratios", default="80/10/10", help="format - [train%]/[valid%]/[test%], set test to 0 to use only old data for testing")
    parser.add_argument("--resample", default=False, action="store_true", help="Include to resample entire dataset into new train/valid/test splits, abstain to retrain against old valid/test")
    parser.add_argument("--new_data_training_threshold", type=int, default=100, help="number of new data samples below which no training occurs")
    parser.add_argument("--interaction_job_num", type=int, default=1000, help="number of dashboard sessions to spin up")
    parser.add_argument(
        "--image_tag", type=str, required=True, help="The tag of docker image that will be used to spin up ecs instance"
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Task name of the ecs instance to be requested"
    )
    opts = parser.parse_args()
    # TODO Implement error handing are argument inputs

    # TODO: parameterize this
    instance_num = opts.interaction_job_num
    
    ij = InteractionJob(instance_num, opts.image_tag, opts.task_name, timeout=IJ_TIMEOUT)
    batch_id = ij.get_batch_id()
    listener = InteractionLogListener(batch_id, IL_TIMEOUT)
    
    ndl = NSPNewDataListener(batch_id=batch_id, opts=opts)

    runner = TaskRunner()
    runner.register_data_generators([ij])
    runner.register_job_listeners([listener])
    runner.register_job_listeners([ndl])
    runner.run()