"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import shutil
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
NSP_RETRAIN_TIMEOUT = 18000  # Wait a max of 5h for the NSP retraining script to finish
MODEL_OUTPUT_POLL_TIME = 60  # Seconds to wait between looking for model output logs

s3 = boto3.client('s3')

MODEL_NAME = "best_model.pth"
MODEL_INFO_NAME = "best_model_info.txt"


class NSPRetrainingJob(DataGenerator):
    def __init__(self):
        super(NSPRetrainingJob, self).__init__()
        self.batch_id = ""

    # Functions originally from sweep_monitor.py:
    def max_and_argmax(self,l):
        i = max(range(len(l)), key=lambda i: l[i])
        return l[i], i

    def get_accs(self):
        accs = {}
        now = datetime.now()
        while not accs:
            if ((datetime.now() - now).seconds > NSP_RETRAIN_TIMEOUT):
                logging.info(f"NSP Retraining has timed out")
                raise TimeoutError
            logging.info(f"Waiting for training logs to appear...")
            time.sleep(MODEL_OUTPUT_POLL_TIME)
            logs = [l for l in os.listdir() if l[-4:] == ".log"]
            for log in logs:
                accs[log] = []
                f = open(log)
                l = f.readlines()
                for line in l:
                    if "evaluating on" in line:
                        s = line.split("\t")
                        a = s[2][11:17]
                        accs[log].append(float(a))
        return accs

    def copy_best_model(self,accs):
        macc = 0.0
        mname = ""
        midx = 0
        for name, acc in accs.items():
            m, mi = self.max_and_argmax(acc)
            if m > macc:
                macc = m
                mname = name
                midx = mi
        source_model = mname[:-4] + "_ep" + str(midx) + ".pth"
        # TODO some sort of lock/semaphore?
        shutil.copyfile(source_model, MODEL_NAME)
        f = open(MODEL_INFO_NAME, "w")
        f.write(source_model + "\n")
        f.write("valid acc " + str(macc)  + "\n")
        f.close()

    def downloadData(self):
        base_data_dir = os.path.join(opts.droidlet_dir, opts.full_data_dir)
        batch_id = str(self.batch_id)
        download_dir = os.path.join(base_data_dir, batch_id) # Currently downloads data to a new dir each time
        try:
            os.mkdir(download_dir)
        except FileExistsError:
            pass
        data_filepath = download_dir + '/annotated.txt'  # Change name on download to match sweep_runner expectations
        try:
            s3.download_file('droidlet-hitl', 'nsp_data.txt', data_filepath)  # Will overwrite file if exists
        except:
            logging.info(f"Exception raised on S3 file download")
            raise
        logging.info(f"New data download completed successfully")
        return batch_id, download_dir


    def run(self):
        logging.info(f"NSP Retraining Job initialized, downloading new data")
        batch_id, download_dir = self.downloadData()
    
        # Setup sweep_runner args
        full_data_dir = download_dir[len(opts.droidlet_dir):] + "/"  # Need to slice off the base droidlet filepath b/c sweep_runner adds it back
        sweep_args = "python3 " + \
            os.path.join(opts.sweep_runner_dir, "sweep_runner.py") + \
            " --sweep_config_folder " + opts.sweep_config_folder + \
            " --sweep_scripts_output_dir " + opts.sweep_scripts_output_dir + \
            " --checkpoint_dir " + opts.checkpoint_dir + \
            " --sweep_name " + batch_id + \
            " --output_dir " + opts.output_dir + \
            " --droidlet_dir " + opts.droidlet_dir + \
            " --full_data_dir " + full_data_dir

        # Initialize the training run
        try:
            sweep = Popen(sweep_args, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        except OSError:
            logging.info(f"Likely error: sweep_runner.py not found where expected")
            raise
        except ValueError:
            logging.info(f"Likely error: Popen called with invalid arguments")
            raise

        # TODO replace with while loop and poll
        try:
            outs, errs = sweep.communicate(timeout=NSP_RETRAIN_TIMEOUT)
            logging.info(f"Sweep successful!")
        except TimeoutExpired:
            sweep.kill()
            outs, errs = sweep.communicate()
            logging.info(f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds")
        logging.info(f"Sweep script outputs: {outs}")
        logging.info(f"Sweep script errors: {errs}")

        # Find the model output directory -- ASSUMES UNIQUE BATCH_ID
        for dir in os.listdir(opts.output_dir):
            if dir.startswith(batch_id):
                output_models_dir = os.path.join(opts.output_dir, (dir + "/"))
        try:
            model_out = os.path.join(output_models_dir, "model_out/")
        except:
            logging.info(f"model output directory not found, check batch ID")
            raise

        # Determine the best model
        os.chdir(model_out)
        accs = self.get_accs()  # Has a built in listener and timeout
        self.copy_best_model(accs)
        
        # Save the best model in S3 bucket
        best_model_path = os.path.join(model_out, "best_model.pth")
        upload_key = batch_id + "/best_model/best_model.pth" 
        s3.upload_file(best_model_path, 'droidlet-hitl', upload_key)

        self.set_finished()
        logging.info(f"NSP Retraining Job finished")


class NSPNewDataListener(JobListener):
    def __init__(self, batch_id):
        super(NSPNewDataListener, self).__init__()
        self.batch_id = batch_id
        self.new_data_found = False

    def run(self, runner):
        logging.info(f"NSP New Data Listener running")

        while not self.check_is_finished():
            time.sleep(LISTENER_SLEEP_TIME)
            try:
                prefix = str(self.batch_id) + '/'
                new_data_key = s3.list_objects_v2(Bucket='droidlet-hitl', Prefix=prefix, Delimiter='/')['Contents'][1]['Key']
                self.new_data_found = True
            except KeyError:
                logging.info(f"New data not yet detected...")
                continue

            # Search for new data
            if (self.new_data_found):
                logging.info(f"NSP Listener has found new data")

                # Initialize retraining job
                nsp_rt = NSPRetrainingJob()
                nsp_rt.batch_id =  self.batch_id # Pass the batch_id to the data generator
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
    parser.add_argument("--checkpoint_dir", default="/checkpoint/aszlam/nsp/")
    opts = parser.parse_args()

    
    ndl = NSPNewDataListener(123)
    runner = TaskRunner()
    runner.register_job_listeners([ndl])
    runner.run()