"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import logging
import os
import shutil
from tabnanny import check
import time
from subprocess import Popen, PIPE
import boto3
from datetime import datetime
import torch
import random

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.artifact_scripts.compute_checksum import compute_checksum_for_directory
from droidlet.tools.artifact_scripts.upload_artifacts_to_aws import (
    compute_checksum,
    tar_and_upload,
    compute_checksum_tar_and_upload,
)
from droidlet.tools.hitl.utils.hitl_recorder import Job, Recorder, JobStat


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
NSP_RETRAIN_TIMEOUT = 72000  # Wait a max of 5h for the NSP retraining script to finish
MODEL_OUTPUT_POLL_TIME = 60  # Seconds to wait between looking for model output logs
NUM_MODEL_OUTPUT_POLLS = 500  # Number of times to poll model output files before assuming finished

s3 = boto3.client("s3")

NUM_TRAINING_RUNS = 6
MODEL_NAME = "best_model.pth"
MODEL_INFO_NAME = "best_model_info.txt"


class NSPRetrainingJob(DataGenerator):
    def __init__(self, job_mng_util: Recorder, batch_id, opts):
        super(NSPRetrainingJob, self).__init__()
        self._job_mng_util = job_mng_util
        self.batch_id = batch_id
        self.opts = opts
        self.exec_training_run = True
        job_mng_util.set_job_stat(Job.RETRAIN, JobStat.ENABLED, True)

    def max_and_argmax(self, l):
        i = max(range(len(l)), key=lambda i: l[i])
        return l[i], i

    def get_accuracy_and_losses(self):
        accs = {}
        losses = {}
        logs = [l for l in os.listdir() if l[-4:] == ".log"]
        if (
            len(logs) < NUM_TRAINING_RUNS
        ):  # Wait until all logs have appeared, takes ~30 min from first to last
            return False
        for log in logs:
            accs[log] = []
            losses[log] = []
            f = open(log)
            l = f.readlines()
            for line in l:
                if "evaluating on" in line:
                    s = line.split("\t")
                    a = s[2][11:17]
                    l = s[1][7:13]
                    accs[log].append(float(a))
                    losses[log].append(float(l))
        return accs, losses

    def copy_best_model(self, accs, losses):
        macc = 0.0
        mname = ""
        midx = 0
        loss = 0.0
        for name, acc in accs.items():
            m, mi = self.max_and_argmax(acc)
            if m >= macc:
                macc = m
                mname = name
                midx = mi
                loss = losses[name][mi]
        source_model = mname[:-4] + "_ep" + str(midx) + ".pth"
        # TODO some sort of lock/semaphore?
        shutil.copyfile(source_model, MODEL_NAME)
        f = open(MODEL_INFO_NAME, "w")
        f.write(source_model + "\n")
        f.write("epoch " + str(midx) + "\n")
        f.write("valid acc " + str(macc) + "\n")
        f.write("valid loss " + str(loss) + "\n")
        f.close()

    def download_data(self, opts, batch_id):
        base_data_dir = os.path.join(opts.droidlet_dir, opts.full_data_dir)
        download_dir = os.path.join(
            base_data_dir, batch_id
        )  # Currently downloads data to a new dir each time
        os.makedirs(download_dir, exist_ok=True)
        data_filepath = download_dir + "/nsp_data.txt"
        try:
            s3.download_file(
                "droidlet-hitl", batch_id + "/nsp_data.txt", data_filepath
            )  # Will overwrite file if exists
        except:
            logging.info(f"Exception raised on S3 data file download")
            self.set_finished(True)
            raise
        logging.info(f"New data download completed successfully")

        # Check the formatting of the file, strip off the index and resave
        annotated_filepath = (
            download_dir + "/annotated.txt"
        )  # Change name to be what sweep_runner expects
        with open(data_filepath, "r") as nspfile, open(annotated_filepath, "w") as afile:
            for line in nspfile:
                parsed = line.split("|")
                if len(parsed) == 2:
                    afile.write(line)
                elif len(parsed) == 3:  # Cut off the index if it exists
                    afile.write(str(parsed[1]) + "|" + str(parsed[2]))
                else:
                    self.set_finished(True)
                    raise ValueError(
                        "Annotated NSP data & logical forms not formatted as expected"
                    )
        logging.info(f"data successfully preprocessed into annotated.txt")

        config_dir = self.create_mask(
            opts=opts, batch_id=batch_id, data_filepath=annotated_filepath
        )

        return download_dir, config_dir

    def create_mask(self, opts, batch_id, data_filepath):
        # Download meta.txt from appropriate S3 bucket
        batch_config_dir = os.path.join(opts.sweep_config_folder, batch_id)
        try:
            os.mkdir(batch_config_dir)
        except FileExistsError:
            pass
        meta_filepath = batch_config_dir + "/meta.txt"
        download_key = batch_id + "/meta.txt"
        logging.info(f"Download key: {download_key}")
        try:
            s3.download_file(
                "droidlet-hitl", download_key, meta_filepath
            )  # Will overwrite file if exists
        except:
            logging.info(f"Exception raised on S3 meta.txt file download")
            self.set_finished(True)
            raise
        logging.info(f"Meta.txt download completed successfully")

        # Copy in sweep config file so mask and config file are in one place
        og_sweep_config = os.path.join(opts.sweep_config_folder, "sweep_config.txt")
        batch_sweep_config = os.path.join(batch_config_dir, "sweep_config.txt")
        shutil.copyfile(og_sweep_config, batch_sweep_config)

        # Pull indices from meta.txt to know which lines of data file should be used for train and valid
        new_data_indices = []
        with open(meta_filepath, "r") as metafile:
            for line in metafile:
                new_data_indices.append(int(line))
        new_data_rows = len(new_data_indices)

        if new_data_rows < opts.new_data_training_threshold:
            logging.warning(
                f"Not enough new data to trigger a training run, one will not be performed"
            )
            self.exec_training_run = False
            return batch_config_dir

        # Create train, valid, and test masks based on user input
        total_rows = sum(1 for line in open(data_filepath))
        logging.info(f"Model training data masks are being generated:")
        old_mask_filepath = os.path.join(opts.sweep_config_folder, "split_masks.pth")
        old_masks_tensor = torch.load(old_mask_filepath)["annotated"]
        old_masks = {
            "train": old_masks_tensor["train"].tolist(),
            "valid": old_masks_tensor["valid"].tolist(),
            "test": old_masks_tensor["test"].tolist(),
        }

        # Set random seed for reproducability and shuffle new mask splits
        # Splits are hard coded as 80% training, 10% validation, 10% test for consistency
        random.seed(batch_id)
        new_masks = (["test"] * int(new_data_rows * 0.1)) + (["valid"] * int(new_data_rows * 0.1))
        new_masks.extend(["train"] * (new_data_rows - len(new_masks)))
        random.shuffle(new_masks)
        # Prepend False for old data
        new_masks[:0] = [False] * (total_rows - new_data_rows)

        all_masks = {"old_masks": old_masks, "new_masks": {}, "both_masks": {}}
        split_keys = ["train", "valid", "test"]
        for mask_type in split_keys:
            # Old masks are static, just extend to be the length of the total dataset
            all_masks["old_masks"][mask_type].extend(
                [False] * (total_rows - len(old_masks[mask_type]))
            )
            # Split the shuffled new splits into three boolean masks
            all_masks["new_masks"][mask_type] = [
                True if x == mask_type else False for x in new_masks
            ]
            # Combine old and new masks for each split and store
            all_masks["both_masks"][mask_type] = [
                True
                if all_masks["old_masks"][mask_type][i] or all_masks["new_masks"][mask_type][i]
                else False
                for i in range(len(new_masks))
            ]

        # Mix and match old and new data based on user input and convert to a tensor
        final_masks = {}
        mask_config_keys = ["old_masks", "new_masks", "both_masks"]
        for i in range(len(opts.retrain_data_splits)):
            final_masks[split_keys[i]] = torch.Tensor(
                all_masks[mask_config_keys[opts.retrain_data_splits[i]]][split_keys[i]]
            ).bool()

        # Report summary stats
        logging.info(f"Percent of data that is new: {((new_data_rows / total_rows)*100):.2f}%")
        logging.info(
            f"Percent of data used for training: {(sum(1 for i in final_masks['train'] if i)*100 / total_rows):.2f}%"
        )
        logging.info(
            f"Percent of data used for validation: {(sum(1 for i in final_masks['valid'] if i)*100 / total_rows):.2f}%"
        )
        logging.info(
            f"Percent of data used for testing: {(sum(1 for i in final_masks['test'] if i)*100 / total_rows):.2f}%"
        )

        # Save nsp retrain job stat
        self._job_mng_util.set_job_stat(
            Job.RETRAIN, JobStat.ORI_DATA_SZ, total_rows - new_data_rows
        )
        self._job_mng_util.set_job_stat(Job.RETRAIN, JobStat.NEW_DATA_SZ, total_rows)

        # Save locally and upload to S3
        mask_filepath = os.path.join(batch_config_dir, "split_masks.pth")
        torch.save({"annotated": final_masks}, mask_filepath)

        upload_key = (
            batch_id
            + f"/split_masks/{opts.retrain_data_splits[0]}_{opts.retrain_data_splits[1]}_{opts.retrain_data_splits[2]}/split_masks.pth"
        )
        response = s3.upload_file(mask_filepath, "droidlet-hitl", upload_key)
        if response:
            logging.info("S3 response: " + response)

        return batch_config_dir

    def slurm_jobs_finished(self, job_ids):
        for job in job_ids:
            cmd = "sacct --jobs={}.batch --format=state".format(job)
            sacct = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
            outs, errs = sacct.communicate(timeout=10)
            outs = outs.replace("\n", "").strip()
            logging.info(f"Slurm status on job {job}: {outs}")
            if "COMPLETED" in outs:
                continue
            elif "FAILED" in outs:
                self.set_finished(True)
                raise RuntimeError("Model retraining failed on SLURM")
            else:
                logging.info(f"Cluster still working, going back to sleep")
                return False
        return True

    def run(self):
        self._job_mng_util.set_job_start(Job.RETRAIN)
        logging.info(f"NSP Retraining Job initialized, downloading new data")
        opts = self.opts

        # Exception handling on user arguments
        if not os.path.isdir(opts.droidlet_dir):
            raise FileNotFoundError("droidlet_dir not found or arg not pathlike")
        if not os.path.isdir(os.path.join(opts.droidlet_dir, opts.full_data_dir)):
            raise FileNotFoundError("full_data_dir not found or arg not pathlike")
        if not os.path.isdir(opts.sweep_runner_dir):
            raise FileNotFoundError("sweep_runner_dir not found or arg not pathlike")
        if not os.path.isdir(opts.sweep_config_folder):
            raise FileNotFoundError("sweep_config_folder not found or arg not pathlike")
        if not os.path.isdir(opts.sweep_scripts_output_dir):
            logging.info(f"Specificed directory for sweep scripts output doesn't exist, build it")
            os.makedirs(opts.sweep_scripts_output_dir, exist_ok=True)
        if not os.path.isdir(opts.output_dir):
            logging.info(f"Specificed directory for output doesn't exist, build it")
            os.makedirs(opts.output_dir, exist_ok=True)
        if not os.path.isdir(opts.checkpoint_dir):
            logging.info(f"Specificed directory for checkpoint doesn't exist, build it")
            os.makedirs(opts.checkpoint_dir, exist_ok=True)
        if opts.new_data_training_threshold < 0:
            raise ValueError("new_data_training_threshold must be >= 0")
        if len(opts.retrain_data_splits) != 3:
            raise TypeError("retrain_data_splits takes exactly three arguments")

        batch_id = str(self.batch_id)
        download_dir, config_dir = self.download_data(opts, batch_id)

        if not self.exec_training_run:
            logging.info(
                f"NSP Retraining Job exiting without retraining model due to insufficient data"
            )
            self.set_finished(True)
            return

        # Setup sweep_runner args
        full_data_dir = download_dir[
            len(opts.droidlet_dir) :
        ]  # Need to slice off the base droidlet filepath b/c sweep_runner adds it back
        sweep_name = (
            batch_id
            + "_mask_opts_"
            + str(opts.retrain_data_splits[0])
            + "_"
            + str(opts.retrain_data_splits[1])
            + "_"
            + str(opts.retrain_data_splits[2])
        )
        sweep_args = (
            "python3 "
            + os.path.join(opts.sweep_runner_dir, "sweep_runner.py")
            + " --sweep_config_folder "
            + config_dir
            + " --sweep_scripts_output_dir "
            + opts.sweep_scripts_output_dir
            + " --checkpoint_dir "
            + opts.checkpoint_dir
            + " --sweep_name "
            + sweep_name
            + " --output_dir "
            + opts.output_dir
            + " --droidlet_dir "
            + opts.droidlet_dir
            + " --full_data_dir "
            + full_data_dir
            + " --pretrained_encoder_name "
            + opts.pretrained_encoder_name
            + " --decoder_config_name "
            + opts.decoder_config_name
            + " --param_update_freq "
            + str(opts.param_update_freq)
        )
        # Require 32g GPU for training
        if opts.use_32g_gpu:
            sweep_args += " --use_32g_gpu "

        # Initialize the training run
        try:
            logging.info(f"Executing in new shell: \n{sweep_args}")
            sweep = Popen(sweep_args, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        except OSError:
            logging.info(f"Likely error: sweep_runner.py not found where expected")
            self.set_finished(True)
            raise
        except ValueError:
            logging.info(f"Likely error: Popen called with invalid arguments")
            self.set_finished(True)
            raise

        # Wait for child process to finish and log outputs/errors
        now = datetime.now()
        while sweep.poll() != 0:
            # TODO: update job completed

            if (datetime.now() - now).seconds > NSP_RETRAIN_TIMEOUT:
                sweep.kill()
                raise TimeoutError(
                    f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds"
                )
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
            self.set_finished(True)
            raise

        # Retrieve cluster job IDs
        job_ids = []
        for line in outs.splitlines():
            if "Submitted" in line:
                job_ids.append(line[-8:])

        # Wait for slurm jobs to complete and determine the best model
        logging.info(f"Watching slurm for job numbers: {job_ids}")
        os.chdir(model_out)
        for i in range(NUM_MODEL_OUTPUT_POLLS):
            if (datetime.now() - now).seconds > NSP_RETRAIN_TIMEOUT:
                raise TimeoutError(
                    f"NSP Retrain child process timed out after {NSP_RETRAIN_TIMEOUT} seconds"
                )
            if self.slurm_jobs_finished(job_ids):
                accs, losses = self.get_accuracy_and_losses()
                self.copy_best_model(accs, losses)
                break
            time.sleep(MODEL_OUTPUT_POLL_TIME)

        # Save the best model in S3 bucket and close out
        upload_key = batch_id + "/best_model/best_model.pth"
        s3.upload_file("best_model.pth", "droidlet-hitl", upload_key)

        # Copy the best model to artifacts folder
        os.chdir(opts.droidlet_dir)
        shutil.copyfile(
            model_out + "best_model.pth",
            "droidlet/artifacts/models/nlu/ttad_bert_updated/caip_test_model.pth",
        )

        # Compute the checksum for model and dataset
        compute_checksum_for_directory("craftassist", "models", "nlu")
        compute_checksum_for_directory("craftassist", "datasets", "")

        # Read checksum of dataset
        checksum_d = ""
        with open("droidlet/tools/artifact_scripts/tracked_checksums/datasets.txt", "r") as f:
            checksum_d = f.read().strip()

        # Write the checksum to local model
        checksum_m, artifact_path_name, artifact_name = compute_checksum(
            "craftassist", "models", "nlu"
        )

        # Log the information for the best model
        os.chdir(self.opts.droidlet_dir)
        with open("droidlet/artifacts/models/nlu/model_log.txt", "w") as f_log, open(
            model_out + MODEL_INFO_NAME, "r"
        ) as f:
            contents = f.readlines()
            epoch = contents[1].split(" ")[1].split("\n")[0]
            acc = contents[2].split(" ")[2].split("\n")[0]
            loss = contents[3].split(" ")[2].split("\n")[0]

            f_log.write("epoch " + epoch + "\n")
            f_log.write("valid_accuracy " + acc + "\n")
            f_log.write("valid_loss " + loss + "\n")
            f_log.write("hash_model " + checksum_m + "\n")
            f_log.write("hash_dataset " + checksum_d + "\n")

            # update corresponding job status
            self._job_mng_util.set_job_stat(Job.RETRAIN, JobStat.MODEL_ACCURACY, acc)
            self._job_mng_util.set_job_stat(Job.RETRAIN, JobStat.MODEL_EPOCH, epoch)
            self._job_mng_util.set_job_stat(Job.RETRAIN, JobStat.MODEL_LOSS, loss)

        # Tar model and upload them to AWS
        tar_and_upload(checksum_m, artifact_path_name, artifact_name)
        # Compute checksum, tar and uploaf for dataset
        compute_checksum_tar_and_upload("craftassist", "datasets", "")

        logging.info(f"NSP Retraining Job finished")

        # TODO: update total finished jobs
        self._job_mng_util.set_job_end(Job.RETRAIN)

        self.set_finished(True)
        return


class NSPNewDataListener(JobListener):
    def __init__(self, batch_id, opts):
        super(NSPNewDataListener, self).__init__()
        self.batch_id = batch_id
        self.new_data_found = False
        self.opts = opts

    def run(self, runner: TaskRunner):
        logging.info(f"NSP New Data Listener running")

        while not self.check_is_finished():
            time.sleep(LISTENER_SLEEP_TIME)
            try:
                prefix = str(self.batch_id) + "/meta.txt"
                new_data_key = s3.list_objects_v2(Bucket="droidlet-hitl", Prefix=prefix)[
                    "Contents"
                ][0]["Key"]
                if new_data_key == prefix:
                    self.new_data_found = True
                else:
                    logging.info(f"New data not yet detected...")
                    continue
            except KeyError:
                logging.info(f"New data not yet detected...")
                continue

            # Search for new data
            if self.new_data_found:
                logging.info(f"NSP Listener has found new data")

                # Initialize retraining job
                nsp_rt = NSPRetrainingJob(
                    job_mng_util=runner.get_job_manage_util(),
                    batch_id=self.batch_id,
                    opts=self.opts,
                )
                runner.register_data_generators([nsp_rt])

                logging.info(f"NSP data gen job registered, listener return")
                self.set_finished(True)
                return


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")

    parser = argparse.ArgumentParser()
    # Retrain pipeline arguments
    parser.add_argument("--batch_id", type=str, default="456", help="Id for this retraining job")
    parser.add_argument(
        "--droidlet_dir", type=str, default=root_dir, help="Relative location of droidlet root"
    )
    parser.add_argument(
        "--full_data_dir",
        default="droidlet/artifacts/datasets/full_data/",
        type=str,
        help="Relative location for data storage",
    )
    parser.add_argument(
        "--sweep_runner_dir",
        type=str,
        default=os.path.join(root_dir, "droidlet/tools/hitl/nsp_retrain/scripts/"),
        help="Relative location of sweep_runner script",
    )
    parser.add_argument(
        "--sweep_config_folder",
        type=str,
        default=os.path.join(root_dir, "droidlet/tools/hitl/nsp_retrain/configs/"),
        help="Relative location of sweep configs",
    )
    parser.add_argument(
        "--sweep_scripts_output_dir",
        type=str,
        default=os.path.join(
            root_dir, "droidlet/tools/hitl/nsp_retrain/outputs/nsp/sweeps/scripts/"
        ),
        help="Relative location for sweep shell scripts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            root_dir, "droidlet/tools/hitl/nsp_retrain/outputs/nsp/sweeps/job_output"
        ),
        help="Relative location for sweep job outputs",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(root_dir, "droidlet/tools/hitl/nsp_retrain/outputs/"),
        help="Relative location of NSP checkpoint folder",
    )
    parser.add_argument(
        "--retrain_data_splits",
        type=int,
        nargs="+",
        choices=[0, 1, 2],
        help="Three int args in the order 'train valid test' where 0=old data only, 1=new data only, 2=all data. Eg. '--retrain_data_splits 2 1 1'",
    )
    parser.add_argument(
        "--new_data_training_threshold",
        default=100,
        type=int,
        help="Number of new data samples below which no training occurs",
    )
    # NLU model arguments
    parser.add_argument(
        "--pretrained_encoder_name",
        default="distilbert-base-uncased",
        type=str,
        help="Pretrained text encoder."
        "See full list at https://huggingface.co/transformers/pretrained_models.html",
    )
    parser.add_argument(
        "--decoder_config_name",
        default="bert-base-uncased",
        type=str,
        help="Name of Huggingface config used to initialize decoder architecture"
        "See full list at https://huggingface.co/transformers/pretrained_models.html",
    )
    # Training hyperparameter
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Number of data per batch",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument("--param_update_freq", default=1, type=int, help="Group N batch updates")
    parser.add_argument(
        "--dtype_samples",
        default="annotated:1.0",
        type=str,
        help="Sampling probabilities for handling different data types",
    )
    parser.add_argument(
        "--use_32g_gpu",
        action="store_true",
        help="Whether use GPU with 32g memory for training",
    )
    parser.add_argument(
        "-decoder_learning_rate", type=float, nargs="+", help="Learning rate for the decoder"
    )
    parser.add_argument(
        "-encoder_learning_rate", type=float, nargs="+", help="Learning rate for the decoder"
    )
    parser.add_argument(
        "--encoder_warmup_steps",
        default=1,
        type=int,
        help="Learning rate warmup steps for the encoder",
    )
    parser.add_argument(
        "--decoder_warmup_steps",
        default=1000,
        type=int,
        help="Learning rate warmup steps for the decoder",
    )
    opts = parser.parse_args()

    # build up sweep config folder if it doesn't exist
    if not os.path.isdir(opts.sweep_config_folder):
        os.makedirs(opts.sweep_config_folder, exist_ok=True)

    # Setup sweep configuration file
    with open(os.path.join(opts.sweep_config_folder, "sweep_config.txt"), "w") as f:
        f.write("batch_size = " + str(opts.batch_size) + "\n")
        f.write("num_epochs = " + str(opts.num_epochs) + "\n")
        f.write("dtype_samples = " + str(opts.dtype_samples) + "\n")
        f.write("decoder_learning_rate = ")
        dlr_list = list(map(str, opts.decoder_learning_rate))
        f.write(", ".join(dlr_list) + "\n")
        f.write("encoder_learning_rate = ")
        elr_list = list(map(str, opts.encoder_learning_rate))
        f.write(", ".join(elr_list) + "\n")

    ndl = NSPNewDataListener(batch_id=opts.batch_id, opts=opts)
    runner = TaskRunner()
    runner.register_job_listeners([ndl])
    runner.run()
