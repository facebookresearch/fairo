"""
Copyright (c) Facebook, Inc. and its affiliates.

This file include a Turk-on-call(TAO) log listener and TAO bug report job data generator for smarter bug report.

The class TaoLogListener(JobListener) listens to the status file for a TAO job batch, 
and initiate a bug report job once logs are ready.

The class TaoBugReportJob(DataGenerator) read in the logs for a TAO job batch,
extract traceback, and output the content of the traceback, the frequency, and associated chat of the traceback to s3. 

"""
import os
import re
import tarfile
import argparse
import shutil
import boto3
import botocore
import logging
import pandas as pd

from droidlet.tools.hitl.data_generator import DataGenerator
from droidlet.tools.hitl.job_listener import JobListener
from droidlet.tools.hitl.task_runner import TaskRunner
from droidlet.tools.hitl.turk_oncall.tests.oncall_job_mock import MockOnCallJob


ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

bucket = s3.Bucket(S3_BUCKET_NAME)
tmp_dir = os.path.join(HITL_TMP_DIR, "tmp")

"""
stat file state 
    - ready:        log generated
    - done:         finished process log job
"""
STAT_READY = "ready"
STAT_DONE = "done"

"""
column names for df & output log csv file
    - content:      traceback content 
    - freq:         count of traceback appear in a batch
"""
COL_CONTENT = "content"
COL_FREQ = "freq"
COL_CHAT = "chat_content"

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)


class TaoBugReportJob(DataGenerator):
    """
    This Data Generator is responsible for spinning up a TAO(Turk-As-Oncall) bug report job.

    It include the following steps:
        1. Unzip log and read
        2. Find Traceback block in the log files
        3. Output traceback to a local temporary file, each batch correspond to one combined traceback report file
        4. Save the traceback report to s3 and clean up local temporary storage
        5. Update the stat file on s3 to "done"

    On a high level:
    - The input of this data generator is a request specifying the batch id for the logs in the corresponding batch to be processed
    - The output of this data generator is a csv file uploaded to s3 which contains the traceback for agent logs in the batch.

    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super().__init__(timeout)
        self._batch_id = batch_id

    def unzip(self, file_path: str, folder_path: str):
        logging.info(f"Tao Bug Report Job] Extracting from compressed file {file_path}")
        file = tarfile.open(file_path)
        file.extractall(folder_path)
        file.close()

    def get_log_traceback(self, path: str, df: pd.DataFrame):
        logging.info(f"Processing log file in {path}")

        # get log files in the path
        for fname in os.listdir(path):
            # only process agent log
            if fname == "agent.log":
                # found a log file
                fpath = os.path.join(path, fname)

                with open(fpath) as file:
                    content = ""
                    chat = ""
                    for line in file:
                        # extract chat
                        if "ttad pre-coref" in line:
                            chat = line

                        # check if starts with YYYY-MM-DD
                        # or line starts with logging level
                        if (
                            re.match(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])", line)
                            or line.startswith("DEBUG")
                            or line.startswith("≈")
                            or line.startswith("WARNING")
                            or line.startswith("ERROR")
                            or line.startswith("CRITICAL")
                        ):
                            # if the content exists & starts with trace back, append to df
                            if content and content.startswith("Traceback"):
                                if content not in df.index:
                                    df.at[content, COL_FREQ] = 0
                                    df.at[content, COL_CHAT] = []
                                df.at[content, COL_FREQ] += 1
                                df.at[content, COL_CHAT].append(chat)
                            content = ""
                        else:
                            content += line

    def run(self) -> None:
        logging.info(f"[Tao Bug Report Job] {self._batch_id} log process started")
        batch_id = self._batch_id
        batch_prefix = f"{batch_id}/interaction/"

        df = pd.DataFrame(columns=[COL_CONTENT, COL_FREQ, COL_CHAT])
        df = df.set_index(COL_CONTENT)

        for obj in bucket.objects.filter(Prefix=batch_prefix):
            # tempory file destination
            dest = os.path.join(tmp_dir, obj.key)

            # get folder path
            folder_path = dest[: dest.rindex("/")]
            os.makedirs(folder_path, exist_ok=True)
            logging.info("Retreiving %s from s3" % obj.key)

            try:
                bucket.download_file(obj.key, dest)
            except botocore.exceptions.ClientError as e:
                logging.info(f"[Tao Bug Report Job] Cannot download {obj.key}")
            else:
                # extract log file
                self.unzip(dest, folder_path)
                # process traceback
                self.get_log_traceback(folder_path, df)

        if len(df) > 0:
            out_remote_path = f"{batch_id}/log_traceback.csv"
            out_local_path = os.path.join(tmp_dir, out_remote_path)

            # Sort and save to s3
            df = df.sort_values(by=COL_FREQ, ascending=False)

            df.to_csv(out_local_path)
            logging.info(
                f"[Tao Bug Report Job] Saving processed log file to s3://{S3_BUCKET_NAME}/{out_remote_path}"
            )

            # save to s3
            try:
                resp = s3.meta.client.upload_file(out_local_path, S3_BUCKET_NAME, out_remote_path)
            except botocore.exceptions.ClientError as e:
                logging.info(f"[TAO Bug Report Job] Not able to save file {out_local_path} to s3.")
        else:
            logging.info(f"[TAO Bug Report Job] Did not find traceback in {batch_id} logs.")

        # delete from local temporary storage
        batch_tmp_path = os.path.join(tmp_dir, f"{batch_id}")
        shutil.rmtree(batch_tmp_path)

        # update status
        stat_fname = f"{batch_id}.stat"
        obj = s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}")

        result = obj.put(Body=STAT_DONE).get("ResponseMetadata")
        if result.get("HTTPStatusCode") == 200:
            self.set_finished(True)
        else:
            logging.info(f"[Tao Bug Report Job] {self._batch_id}.stat not updated")


class TaoLogListener(JobListener):
    """
    This Listener is responsible for listening to new generated log for a TAO job batch.

    The TaoLogListener constructor takes a batch_id input, which specified the batch the listener is associated with.

    The steps are:
        1. Check if there s3 for a stat file in the batch folder
        2. If the stat file indicate the logs are ready, initiate a bug report job

    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(TaoLogListener, self).__init__(timeout=timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
        batch_id = self._batch_id

        while not self.check_is_finished():
            logging.info(f"[TAO Log Listener] Checking status for {batch_id}...")
            stat_fname = f"{batch_id}.stat"
            finished = True

            # check if stat file exist
            try:
                s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}").load()
            except botocore.exceptions.ClientError as e:
                logging.info(
                    f"[TAO Log Listener] No new data for {batch_id} ... Remaining time: {self.get_remaining_time()}"
                )
            else:
                # read stat
                resp = s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}").get()
                stat = resp["Body"].read().decode("utf-8")

                if stat == STAT_READY:
                    # create a tao bug report job
                    tlo_job = TaoBugReportJob(batch_id=batch_id)
                    runner.register_data_generators([tlo_job])
                else:
                    logging.info(
                        f"[TAO Log Listener] Status for {batch_id} is not {STAT_READY}, but is {stat}"
                    )

            if not self.check_parent_finished():
                finished = False
            self.set_finished(finished)


def process_past_logs(batch_ids: list):
    """
    process past logs in every batch listed in batch ids
    """
    runner = TaskRunner()

    for batch_id in batch_ids:
        mock_data_generator = MockOnCallJob(batch_id)
        runner.register_data_generators([mock_data_generator])
        tao_log_listener = TaoLogListener(batch_id=batch_id)
        tao_log_listener.add_parent_jobs([mock_data_generator])
        runner.register_job_listeners([tao_log_listener])
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--tao_job_batch_ids",
        type=int,
        nargs="+",
        required=True,
        help="TAO job batch ids",
    )
    opts = parser.parse_args()
    print(opts.tao_job_batch_ids)
    process_past_logs(opts.tao_job_batch_ids)
