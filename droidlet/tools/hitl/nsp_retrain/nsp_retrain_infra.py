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
import torch

from typing import List

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
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

LISTENER_SLEEP_TIME = 10  # Seconds to wait before listener looks for new data
NSP_RETRAIN_TIMEOUT = 18000  # Wait a max of 5h for the NSP retraining script to finish
MODEL_OUTPUT_POLL_TIME = 60  # Seconds to wait between looking for model output logs
NUM_MODEL_OUTPUT_POLLS = 500  # Number of times to poll model output files before assuming finished

s3 = boto3.client('s3')

NUM_TRAINING_RUNS = 6
MODEL_NAME = "best_model.pth"
MODEL_INFO_NAME = "best_model_info.txt"


class NSPRetrainingJob(DataGenerator):
    def __init__(self, batch_id, opts):
        super(NSPRetrainingJob, self).__init__()
        self.batch_id = batch_id
        self.opts = opts
        self.exec_training_run = True


    def max_and_argmax(self,l):
        i = max(range(len(l)), key=lambda i: l[i])
        return l[i], i


    def get_accs(self):
        accs = {}
        logs = [l for l in os.listdir() if l[-4:] == ".log"]
        if len(logs) < NUM_TRAINING_RUNS:  # Wait until all logs have appeared, takes ~30 min from first to last
            return False
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


    def download_data(self, opts, batch_id):
        base_data_dir = os.path.join(opts.droidlet_dir, opts.full_data_dir)
        download_dir = os.path.join(base_data_dir, batch_id) # Currently downloads data to a new dir each time
        try:
            os.mkdir(download_dir)
        except FileExistsError:
            pass
        data_filepath = download_dir + '/nsp_data.txt'
        try:
            s3.download_file('droidlet-hitl', 'nsp_data.txt', data_filepath)  # Will overwrite file if exists
        except:
            logging.info(f"Exception raised on S3 data file download")
            raise
        logging.info(f"New data download completed successfully")

        # Check the formatting of the file, strip off the index and resave
        annotated_filepath = download_dir + '/annotated.txt'  # Change name to be what sweep_runner expects
        with open(data_filepath, "r") as nspfile, open(annotated_filepath, "w") as afile:
            for line in nspfile:
                parsed = line.split('|')
                if len(parsed) == 2:
                    afile.write(line)
                elif len(parsed) == 3:  # Cut off the index if it exists
                    afile.write(str(parsed[1]) + '|' + str(parsed[2]))
                else:
                    raise ValueError("Annotated NSP data & logical forms not formatted as expected")
        logging.info(f"data successfully preprocessed into annotated.txt")

        config_dir = self.create_mask(opts=opts, batch_id=batch_id, data_filepath=annotated_filepath)

        return download_dir, config_dir


    def create_mask(self, opts, batch_id, data_filepath):
        # Download meta.txt from appropriate S3 bucket
        batch_config_dir = os.path.join(opts.sweep_config_folder, batch_id)
        try:
            os.mkdir(batch_config_dir)
        except FileExistsError:
            pass
        meta_filepath = batch_config_dir + '/meta.txt'
        download_key = batch_id + '/meta.txt'
        logging.info(f"Download key: {download_key}")
        try:
            s3.download_file('droidlet-hitl', download_key, meta_filepath)  # Will overwrite file if exists
        except:
            logging.info(f"Exception raised on S3 meta.txt file download")
            raise
        logging.info(f"Meta.txt download completed successfully")

        # Copy in sweep config file so mask and config file are in one place
        og_sweep_config = os.path.join(opts.sweep_config_folder, 'sweep_config.txt')
        batch_sweep_config = os.path.join(batch_config_dir, 'sweep_config.txt')
        shutil.copyfile(og_sweep_config, batch_sweep_config)

        # Pull indices from meta.txt to know which lines of data file should be used for train and valid
        new_data_indices = []
        with open(meta_filepath, "r") as metafile:
            for line in metafile:
                new_data_indices.append(int(line))
        new_data_rows = len(new_data_indices)

        if (new_data_rows < int(opts.new_data_training_threshold)):
            logging.warning(f"Not enough new data to trigger a training run, one will not be performed")
            self.exec_training_run = False

        # Import split ratios (default) 80% of new data for training, 10% for validation, and 10% for testing
        # TODO handle formatting errors, negative numbers (here or in arg parser?)
        data_split = opts.data_split_ratios.split('/')
        data_split = [int(x)/100 for x in data_split]  # Convert from str % to fraction
        logging.info(f"Data split ratios - train: {data_split[0]}, valid: {data_split[1]}, test: {data_split[2]}")
        if (sum(data_split) != 1):
            raise ValueError("Data splits do not sum to 100%")

        # Randomize the submasks for train, valid, and test
        total_rows = sum(1 for line in open(data_filepath))
        train_submask = torch.rand(new_data_rows) < data_split[0]  # Approx. % of new data used for training
        valid_submask = torch.rand(new_data_rows) < data_split[1]  # Approx. % of new data used for validation
        if (data_split[2] > 0):  # Use equal parts new and old data for testing
            test_submask = torch.rand(total_rows) < ((new_data_rows/(total_rows - new_data_rows)) * data_split[2]) 
        else:  # Use only old data, approx. the same number of samples as in valid
            test_submask = torch.rand(total_rows) < ((new_data_rows/(total_rows - new_data_rows)) * data_split[1])

        # Generate the full masks that are the same length as the input data
        train_mask, valid_mask, test_mask = [], [], []
        submask_pointer = 0
        for i in range(total_rows):
            if i in new_data_indices:
                if train_submask[submask_pointer]:
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                elif valid_submask[submask_pointer]:
                    train_mask.append(False)
                    valid_mask.append(True)
                    test_mask.append(False)
                else:
                    train_mask.append(False)
                    valid_mask.append(False)
                    test_mask.append(True)  # All new data that's not in train or valid is used for test
                submask_pointer += 1
            else:
                train_mask.append(False)  # No old data used for train or valid
                valid_mask.append(False)
                if test_submask[i]:
                    test_mask.append(True)
                else:
                    test_mask.append(False)
        perc_new = new_data_rows / total_rows
        perc_train = sum(1 for i in train_mask if i) / total_rows
        perc_valid = sum(1 for i in valid_mask if i) / total_rows
        perc_test = sum(1 for i in test_mask if i) / total_rows
        logging.info(f"Fraction of data that is new: {perc_new}")
        logging.info(f"Actual fraction of data used for training (~{data_split[0]*100}% of new): {perc_train}")
        logging.info(f"Actual fraction of data used for validation (~{data_split[1]*100}% of new): {perc_valid}")
        logging.info(f"Actual fraction of data used for testing: {perc_test}")

        #reformat as dict with the appropriate keys and save
        train_mask = torch.Tensor(train_mask).bool()
        valid_mask = torch.Tensor(valid_mask).bool()
        test_mask = torch.Tensor(test_mask).bool()
        mask_dict = {'annotated': {'train': train_mask, 'valid': valid_mask, 'test': test_mask}}
        logging.info(f"Mask dictionary: {mask_dict}")
        mask_filepath = batch_config_dir + '/split_masks.pth'
        torch.save(mask_dict, mask_filepath)

        return batch_config_dir


    def slurm_jobs_finished(self, job_ids):
        for job in job_ids:
            cmd = 'sacct --jobs={}.batch --format=state'.format(job)
            sacct = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
            outs, errs = sacct.communicate(timeout=10)
            outs = outs.replace('\n','').strip()
            logging.info(f"Slurm status on job {job}: {outs}")
            if 'COMPLETED' in outs:
                continue
            else:
                logging.info(f"Cluster still working, going back to sleep")
                return False
        return True


    def run(self):
        logging.info(f"NSP Retraining Job initialized, downloading new data")
        opts = self.opts
        batch_id = str(self.batch_id)
        download_dir, config_dir = self.download_data(opts, batch_id)

        if not self.exec_training_run:
            logging.info(f"NSP Retraining Job exiting without retraining model due to insufficient data")
            self.set_finished(True)
            return
    
        # Setup sweep_runner args
        full_data_dir = download_dir[len(opts.droidlet_dir):]  # Need to slice off the base droidlet filepath b/c sweep_runner adds it back
        sweep_args = "python3 " + \
            os.path.join(opts.sweep_runner_dir, "sweep_runner.py") + \
            " --sweep_config_folder " + config_dir + \
            " --sweep_scripts_output_dir " + opts.sweep_scripts_output_dir + \
            " --checkpoint_dir " + opts.checkpoint_dir + \
            " --sweep_name " + batch_id + \
            " --output_dir " + opts.output_dir + \
            " --droidlet_dir " + opts.droidlet_dir + \
            " --full_data_dir " + full_data_dir

        # Initialize the training run
        try:
            logging.info(f"Executing in new shell: \n{sweep_args}")
            sweep = Popen(sweep_args, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        except OSError:
            logging.info(f"Likely error: sweep_runner.py not found where expected")
            raise
        except ValueError:
            logging.info(f"Likely error: Popen called with invalid arguments")
            raise

        # Wait for child process to finish and log outputs/errors
        now = datetime.now()
        while (sweep.poll() != 0):
            if ((datetime.now() - now).seconds > NSP_RETRAIN_TIMEOUT):
                sweep.kill()
                raise TimeoutError(f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds")
                break
        outs, errs = sweep.stdout.read(), sweep.stderr.read()
        logging.info(f"Sweep script outputs: \n{outs}")
        logging.info(f"Sweep script errors: \n{errs}")

        # Find the model output directory -- ASSUMES UNIQUE BATCH_ID
        for dir in os.listdir(opts.output_dir):
            if dir.startswith(batch_id):
                output_models_dir = os.path.join(opts.output_dir, (dir + "/"))
        try:
            model_out = os.path.join(output_models_dir, "model_out/")
        except:
            logging.info(f"model output directory not found, check batch ID")
            raise

        # Retrieve cluster job IDs
        job_ids = []
        for line in outs.splitlines():
            if 'Submitted' in line:
                job_ids.append(line[-8:])

        # Wait for slurm jobs to complete and determine the best model
        logging.info(f"Watching slurm for job numbers: {job_ids}")
        os.chdir(model_out)
        for i in range(NUM_MODEL_OUTPUT_POLLS):
            if ((datetime.now() - now).seconds > NSP_RETRAIN_TIMEOUT):
                raise TimeoutError(f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds")
            if self.slurm_jobs_finished(job_ids):
                accs = self.get_accs()
                self.copy_best_model(accs)
                break
            time.sleep(MODEL_OUTPUT_POLL_TIME)

        # Save the best model in S3 bucket and close out
        upload_key = batch_id + "/best_model/best_model.pth" 
        s3.upload_file("best_model.pth", 'droidlet-hitl', upload_key)

        logging.info(f"NSP Retraining Job finished")
        self.set_finished(True)
        return


class NSPNewDataListener(JobListener):
    def __init__(self, batch_id, opts, timeout=-1):
        super(NSPNewDataListener, self).__init__(timeout=timeout)
        self.batch_id = batch_id
        self.new_data_found = False
        self.opts = opts

    def run(self, runner):
        logging.info(f"NSP New Data Listener running")

        while not self.check_is_finished():
            time.sleep(LISTENER_SLEEP_TIME)
            try:
                prefix = str(self.batch_id) + '/meta.txt'
                new_data_key = s3.list_objects_v2(Bucket='droidlet-hitl', Prefix=prefix, Delimiter='/')['Contents'][0]['Key']
                self.new_data_found = True
            except KeyError:
                logging.info(f"[NSP Data Listener] New data not yet detected...Remaining time: {self.get_remaining_time()}")
                continue

            # Search for new data
            if (self.new_data_found):
                logging.info(f"NSP Listener has found new data")

                # Initialize retraining job
                nsp_rt = NSPRetrainingJob(batch_id=self.batch_id, opts=self.opts)
                runner.register_data_generators([nsp_rt])

                logging.info(f"NSP data gen job registered, listener return")
                self.set_finished(True)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droidlet_dir", default="/private/home/yuxuans/Workspace/droidlet/")
    parser.add_argument("--full_data_dir", default="agents/craftassist/datasets/full_data/")
    parser.add_argument("--sweep_runner_dir", default="/checkpoint/yuxuans/nsp_cl_scripts/")
    parser.add_argument("--sweep_config_folder", default="/checkpoint/yuxuans/nsp/sweeps/scripts/configs/auto_sweep_configs/")
    parser.add_argument("--sweep_scripts_output_dir", default="/checkpoint/yuxuans/nsp/sweeps/scripts/")
    parser.add_argument("--output_dir", default="/checkpoint/yuxuans/nsp/sweeps/job_output/")
    parser.add_argument("--checkpoint_dir", default="/checkpoint/yuxuans/nsp/")
    parser.add_argument("--data_split_ratios", default="80/10/10", help="format - [train%]/[valid%]/[test%], set test to 0 to use only old data for testing")
    parser.add_argument("--new_data_training_threshold", default="100", help="number of new data samples below which no training occurs")
    opts = parser.parse_args()
    # TODO Implement error handing are argument inputs

    
    ndl = NSPNewDataListener(batch_id=991, opts=opts)
    runner = TaskRunner()
    runner.register_job_listeners([ndl])
    runner.run()