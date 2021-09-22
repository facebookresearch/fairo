"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import re
import time
import subprocess
import boto3
from datetime import datetime, date

from typing import List

from utils.data_generator import DataGenerator
from utils.job_listener import JobListener
from utils.task_runner import TaskRunner


log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)

LISTENER_SLEEP_TIME = 10  # Number of seconds to wait before looking for new data

s3 = boto3.client('s3')


class NSPRetrainingJob(DataGenerator):
    def __init__(self):
        super(NSPRetrainingJob, self).__init__()
        self.data_datetime = ""

    def run(self):
        logging.info(f"NSP Retraining Job initialized")

        # Currently the sweep_runner script uses local data, unclea how to pass in data from S3
        #with open('filename', 'wb') as data:
        #    s3.download_fileobj('mybucket', 'mykey', data)

        sweep_filepath = os.path.join(opts.sweep_runner_dir, "sweep_runner.py")
        sweep_params = " \
            --sweep_config_folder " + opts.sweep_config_folder + " \
            --sweep_scripts_output_dir " + opts.sweep_scripts_output_dir + " \
            --sweep_name " + opts.sweep_name + " \
            --output_dir " + opts.output_dir

        # .run() by default will wait until the child process is complete and CompletedProcess is returned
        sweep = subprocess.run(sweep_filepath, input=sweep_params, text=True)  # Shaky on the details, check other args      
        sweep_successful = sweep.returncode      
        sweep_output = sweep.stdout

        now = datetime.now()
        nowstr = now.strftime("_%m_%d_%H_%M")
        # Needs to be resilient to --append_date option being false, or we could force a new job name each time


        output_models_dir = os.path.join(opts.output_dir, opts.sweep_name)
        #best_model = 

        #  1) Initialize training run, pointing the script to appropriate data
            # TODO figure out how to point the sweep_runner script to S3 data
        #  2) Wait for run to finish (look for flag?)                               -done automatically
        #  3) Select optimal output model from sweep
        #  4) Save the updated model in the appropriate place

        self.set_finished()
        logging.info(f"NSP Retraining Job finished")


class NSPNewDataListener(JobListener):
    def __init__(self):
        super(NSPNewDataListener, self).__init__()

    def retrieveMostRecentData(self):
        runs_dict = s3.list_objects_v2(Bucket='craftassist', Prefix='turk_interactions_with_agent/', Delimiter='/')["CommonPrefixes"]
        run_times = [x['Prefix'].split('/')[1] for x in runs_dict]
        cleaned_times = [x for x in run_times if len(x) == 32]  # A bit sloppy
        cleaned_times.sort()  # List comes sorted by time, so this is just defensive
        return cleaned_times[-1]

    def run(self, runner):
        logging.info(f"NSP New Data Listener running")
        data_checkpoint = self.retrieveMostRecentData()

        while not self.check_is_finished():
            finished = True
            time.sleep(LISTENER_SLEEP_TIME)
            
            #  1) Initialize concept of "old" data as the most recent run at script start
                # TODO Change this to be persistent storage on S3?
            #  2) Search for new data
            #    -If none found, go back to sleep/listening
            #  3) Package new data in the format required by the training script
            #  4) Launch a new DataGenerator job
            #  5) Go back to sleep/listening

            most_recent = self.retrieveMostRecentData()
            if (most_recent > data_checkpoint):
                nsp_rt = NSPRetrainingJob()
                nsp_rt.data_datetime = most_recent  # TODO Might not actually need to do this
                self.add_parent_jobs(nsp_rt)
                runner.register_data_generators([nsp_rt])

                # TODO need to re-run the task runner?

                continue

            if not self.check_parent_finished():
                finished = False

            # A bit unclear what the exit condition should be here.  Are we listening forever?
            self.set_finished(finished)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_runner_dir", default="/checkpoint/aszlam/nsp_cl_scripts/")
    parser.add_argument("--sweep_config_folder", default="/checkpoint/aszlam/nsp/sweeps/scripts/configs/auto_sweep_configs/")
    parser.add_argument("--sweep_scripts_output_dir", default="/checkpoint/aszlam/nsp/sweeps/scripts/")
    parser.add_argument("--sweep_name", default="auto")
    parser.add_argument("--output_dir", default="/checkpoint/aszlam/nsp/sweeps/job_output/")
    opts = parser.parse_args()

    
    ndl = NSPNewDataListener()
    runner = TaskRunner()
    runner.register_job_listeners([ndl])
    runner.run()