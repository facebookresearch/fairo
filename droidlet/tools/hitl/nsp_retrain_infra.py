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
        self.data_dir = ""

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
            # Make sure to set a timeout (a few hours), implement error handling
            # Use popen to run async, and check for completion    
        sweep_successful = sweep.returncode      
        sweep_output = sweep.stdout

        now = datetime.now()
        nowstr = now.strftime("_%m_%d_%H_%M")
        # Needs to be resilient to --append_date option being false, or we could force a new job name each time

        if (opts.append_date):
            pass
        else:
            output_models_dir = os.path.join(opts.output_dir, opts.sweep_name)

        
        #best_model = 

        #  1) Initialize training run, pointing the script to appropriate data
            # TODO figure out how to point the sweep_runner script to S3 data
        #  2) Wait for run to finish (look for flag?)                               -done automatically
        #  3) Select optimal output model from sweep
        #  4) Save the best model in S3 bucket

        # Listener should pass S3 filepath to data generator, which downloads the data to the specified local dir, where it's referenced by the sweep

        self.set_finished()
        logging.info(f"NSP Retraining Job finished")


class NSPNewDataListener(JobListener):
    def __init__(self):
        super(NSPNewDataListener, self).__init__()

    def retrieveMostRecentData(self):
        # TODO Is this the right place?  Yuxuan is going to upload some dummy data.
        runs_dict = s3.list_objects_v2(Bucket='craftassist', Prefix='turk_interactions_with_agent/', Delimiter='/')["CommonPrefixes"]
        run_times = [x['Prefix'].split('/')[1] for x in runs_dict]
        cleaned_times = [x for x in run_times if len(x) == 32]  # A bit sloppy...all datetimes are currently 32 chars
        cleaned_times.sort()  # List comes sorted by time, so this is just defensive
        return cleaned_times[-1]

    def run(self, runner):
        logging.info(f"NSP New Data Listener running")

        # Initialize concept of "old" data as the most recent at runtime
            # TODO Change this to be persistent storage on S3?
        data_checkpoint = self.retrieveMostRecentData()

        while not self.check_is_finished():
            time.sleep(LISTENER_SLEEP_TIME)            

            # Search for new data
            most_recent = self.retrieveMostRecentData()
            if (most_recent > data_checkpoint):
                logging.info(f"NSP Listener has found new data")

                # Download the new data to local dir
                full_data_dir = os.path.join(opts.droidlet_dir, opts.full_data_dir)
                download_dir = os.path.join(full_data_dir, most_recent)
                os.mkdir(download_dir)
                data_filepath = download_dir + '/logs.tar.gz'
                download_key = 'turk_interactions_with_agent/' + most_recent + '/logs.tar.gz'
                try:
                    s3.download_file('craftassist', download_key, data_filepath)
                except:
                    logging.info(f"Exception raised on S3 file download")
                    raise

                # TODO Does the tarball need to be unpacked?  There's not an obvious txt file inside.

                # Initialize retraining job
                nsp_rt = NSPRetrainingJob()
                nsp_rt.data_dir =  data_filepath # Pass the new data local path to the data generator
                    # TODO double check
                # self.add_parent_jobs(nsp_rt)
                runner.register_data_generators([nsp_rt])

                logging.info(f"NSP data gen job registered, listener return")
                self.set_finished(True)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/private/home/aszlam/fairinternal/droidlet/")
    parser.add_argument("--full_data_dir", default="agents/craftassist/datasets/full_data/")
    parser.add_argument("--sweep_runner_dir", default="/checkpoint/aszlam/nsp_cl_scripts/")
    parser.add_argument("--sweep_config_folder", default="/checkpoint/aszlam/nsp/sweeps/scripts/configs/auto_sweep_configs/")
    parser.add_argument("--sweep_scripts_output_dir", default="/checkpoint/aszlam/nsp/sweeps/scripts/")
    parser.add_argument("--sweep_name", default="auto")
    parser.add_argument("--output_dir", default="/checkpoint/aszlam/nsp/sweeps/job_output/")
    parser.add_argument("--append_date", action="store_false")
    opts = parser.parse_args()

    
    ndl = NSPNewDataListener()
    runner = TaskRunner()
    runner.register_job_listeners([ndl])
    runner.run()