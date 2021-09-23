"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import time
from subprocess import Popen, PIPE, TimeoutExpired
import boto3
from datetime import datetime

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

LISTENER_SLEEP_TIME = 10  # Seconds to wait before listener looks for new data
NSP_RETRAIN_TIMEOUT = 36000  # Wait a max of 10h for the NSP retraining script to finish

s3 = boto3.client('s3')


class NSPRetrainingJob(DataGenerator):
    def __init__(self):
        super(NSPRetrainingJob, self).__init__()
        self.data_prefix = ""

    def run(self):
        logging.info(f"NSP Retraining Job initialized, downloading new data")

        # Download the new data to local dir
        data_dir = os.path.join(opts.droidlet_dir, opts.full_data_dir)
        tar_download_dir = os.path.join(data_dir, self.data_prefix)
        os.mkdir(tar_download_dir)
        tar_filepath = tar_download_dir + '/logs.tar.gz'
        download_key = 'turk_interactions_with_agent/' + self.data_prefix + '/logs.tar.gz'
        try:
            s3.download_file('craftassist', download_key, tar_filepath)
        except:
            logging.info(f"Exception raised on S3 file download")
            raise
        logging.info(f"New data download completed successfully")
        # TODO Does the tarball need to be unpacked?  There's not an obvious txt file inside.

        full_data_dir = tar_download_dir[len(opts.droidlet_dir):]  # Need to slice off the base droidlet filepath b/c sweep_runner adds it back
        # Recommended to pass args to Popen as a single string if shell=True
        sweep_args = "python3 " + \
            os.path.join(opts.sweep_runner_dir, "sweep_runner.py ") + \
            "--sweep_config_folder " + opts.sweep_config_folder + \
            "--sweep_scripts_output_dir " + opts.sweep_scripts_output_dir + \
            "--sweep_name " + opts.sweep_name + \
            "--output_dir " + opts.output_dir + \
            "--droidlet_dir " + opts.droidlet_dir + \
            "--full_data_dir " + full_data_dir

        # Initialize the training run
        try:
            sweep = Popen(sweep_args, shell=True, stdout=PIPE, stderr=PIPE, text=True)
                # Use env to set any environment vars
        except OSError:
            logging.info(f"Likely error: sweep_runner.py not found where it should be")
            raise
        except ValueError:
            logging.info(f"Likely error: Popen called with invalid arguments")
            raise

        try:
            outs, errs = sweep.communicate(timeout=NSP_RETRAIN_TIMEOUT)
                # TODO Figure out how blocking this is and what is acceptable
            logging.info(f"Sweep successful!")
            logging.info(f"Sweep script outputs: {outs}")
            logging.info(f"Sweep script errors: {errs}")
        except TimeoutExpired:
            sweep.kill()
            outs, errs = sweep.communicate()
            logging.info(f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds")
            logging.info(f"Sweep script outputs: {outs}")
            logging.info(f"Sweep script errors: {errs}")

        if (opts.append_date):
            # If we assume that sweep_name isn't unique, instead we should look for the most recent run agnostic to the time it is now
            runs = [d for d in os.listdir(opts.output_dir) if os.path.isdir(os.path.join(opts.output_dir, d))]
            run_times = [run[-12:] for run in runs if len(run) >= 12]  # Filter for just the run time suffixes
                # TODO makes a waak assumption about what types of directory names will be present
            run_times.sort()

            for dir in os.listdir(opts.ouput_dir):
                if dir.endswith(run_times[-1]):
                    output_models_dir = os.path.join(opts.output_dir, (dir + "/"))
                    # Slightly vulnerable to a bug where this doesn't trigger
        else:
            output_models_dir = os.path.join(opts.output_dir, (opts.sweep_name + "/"))

        # Retrieve the best model
        model_out = os.path.join(output_models_dir, "model_out/")
        best_model_info = os.path.join(model_out, "best_model_info.txt")
        with open(best_model_info, "r") as f:
            best_model_name = f.readline()
        best_model_name = best_model_name[:-2]  # Chop off newline chars
        
        # Save the best model in S3 bucket
        best_model_path = os.path.join(model_out, best_model_name)
        s3.upload_file(best_model_path, 'craftassist', UPLOAD_KEY)
            # TODO replace with actual upload key

        self.set_finished()
        logging.info(f"NSP Retraining Job finished")


class NSPNewDataListener(JobListener):
    def __init__(self, batch_id):
        super(NSPNewDataListener, self).__init__()
        self.batch_id = batch_id

    def retrieveMostRecentData(self):
        # TODO Is this the right place?  Yuxuan is going to upload some dummy data.
        runs_dict = s3.list_objects_v2(Bucket='droidlet-hitl', Prefix='123/', Delimiter='/')["CommonPrefixes"]
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

                # Initialize retraining job
                nsp_rt = NSPRetrainingJob()
                nsp_rt.data_prefix =  most_recent # Pass the new data prefix to the data generator
                # self.add_parent_jobs(nsp_rt)
                runner.register_data_generators([nsp_rt])

                logging.info(f"NSP data gen job registered, listener return")
                self.set_finished(True)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/private/home/aszlam/fairinternal/droidlet/")
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