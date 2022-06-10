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


ECS_INSTANCE_TIMEOUT = 45
INTERACTION_JOB_POLL_TIME = 30
INTERACTION_LISTENER_POLL_TIME = 30
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

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

COL_CONTENT = "content"
COL_FREQ = "freq"

"""
stat file state 
    - ready:        log generated
    - done:         finished process log job
"""
STAT_READY = "ready"
STAT_DONE = "done"

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel("INFO")
sh = logging.StreamHandler()
sh.setFormatter(log_formatter)
logger.addHandler(sh)


class TaoLogOutputJob(DataGenerator):
    """Process tao log and output:
    1. Unzip log and read
    2. Find Traceback
    3. Output traceback to a file
    4. Save on s3
    """

    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super().__init__(timeout)
        self._batch_id = batch_id

    def unzip(self, file_path: str, folder_path: str):
        logging.info(f"Tao Log Output Job] Extracting from compressed file {file_path}")
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
                    for line in file:
                        # check if starts with YYYY-MM-DD
                        # or line starts with logging level
                        if (
                            re.match(r"^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])", line)
                            or line.startswith("DEBUG")
                            or line.startswith("INFO")
                            or line.startswith("WARNING")
                            or line.startswith("ERROR")
                            or line.startswith("CRITICAL")
                        ):
                            # if the content exists & starts with trace back, append to df
                            if content and content.startswith("Traceback"):
                                if content not in df.index:
                                    df.loc[content] = 0
                                df.loc[content] += 1
                            content = ""
                        else:
                            content += line

    def run(self) -> None:
        logging.info(f"[Tao Log Output Job] {self._batch_id} log process started")
        batch_id = self._batch_id
        batch_prefix = f"{batch_id}/interaction/"

        df = pd.DataFrame(columns=[COL_CONTENT, COL_FREQ])
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
                logging.info(f"[Tao Log Output Job] Cannot download {obj.key}")
            else:
                # extract log file
                self.unzip(dest, folder_path)
                # process traceback
                self.get_log_traceback(folder_path, df)

        if len(df) > 0:
            out_remote_path = f"{batch_id}/log_traceback.csv"
            out_local_path = os.path.join(tmp_dir, out_remote_path)

            # Dedup based on content column and save
            df.to_csv(out_local_path)
            logging.info(
                f"[Tao Log Output Job] Saving processed log file to s3://{S3_BUCKET_NAME}/{out_remote_path}"
            )

            # save to s3
            try:
                resp = s3.meta.client.upload_file(out_local_path, S3_BUCKET_NAME, out_remote_path)
            except botocore.exceptions.ClientError as e:
                logging.info(f"[TAO Log Listener] Not able to save file {out_local_path} to s3.")

        # delete from local temporary storage
        batch_tmp_path = os.path.join(tmp_dir, f"{batch_id}")
        shutil.rmtree(batch_tmp_path)

        # update status
        stat_fname = f"{batch_id}.stat"
        obj = s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}")

        result = obj.put(Body=STAT_DONE).get("ResponseMetadata")
        if result.get("HTTPStatusCode") == 200:
            self.set_finished()
        else:
            logging.info(f"[Tao Log Output Job] {self._batch_id}.stat not updated")


class TaoLogListener(JobListener):
    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super(TaoLogListener, self).__init__(timeout=timeout)
        self._batch_id = batch_id

    def run(self, runner: TaskRunner) -> None:
        batch_id = self._batch_id

        while not self.check_is_finished():
            logging.info(f"[TAO Log Listener] Checking status for {batch_id}...")
            stat_fname = f"{batch_id}.stat"

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
                    # create a tao log output job
                    tlo_job = TaoLogOutputJob(batch_id=batch_id)
                    runner.register_data_generators([tlo_job])

            self.set_finished(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tao_job_batch_id",
        type=int,
        required=True,
        help="TAO job batch id",
    )
    opts = parser.parse_args()

    runner = TaskRunner()

    # test on hard coded batch id
    tao_log_listener = TaoLogListener(batch_id=opts.tao_job_batch_id)
    runner.register_job_listeners([tao_log_listener])

    runner.run()
