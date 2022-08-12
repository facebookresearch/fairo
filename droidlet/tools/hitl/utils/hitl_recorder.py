"""
Copyright (c) Facebook, Inc. and its affiliates.
This file include job management utils for recording hitl job run statistics,
including meta data like name, batch id, and job specific statistics like 
failed/sucess command in interaction jobs, model accuracy in the nsp retrain jobs etc.
This util class is extendable, if you want to add a new statistics for recording, 
please do the following:
    - To add a new meta data - please add a new entry in the MetaData enum.
    - To add a new job type - please add a new entry in the Job enum.
    - To add a new job statistic - please add a new entry in the JobStat enum.
"""
import logging
import boto3
import botocore
import datetime
import json
from enum import Enum
import os

# for s3
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

# local tmp directory
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

# job management record path prefix
JOB_MNG_PATH_PREFIX = "job_management_records"

s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

bucket = s3.Bucket(S3_BUCKET_NAME)


class MetaData(Enum):
    BATCH_ID = "batch_id"
    NAME = "name"
    S3_LINK = "s3_link"
    COST = "cost"
    START_TIME = "start_time"
    END_TIME = "end_time"


class Job(Enum):
    INTERACTION = "interaction"
    ANNOTATION = "annotation"
    RETRAIN = "retrain"


class JobStat(Enum):
    ENABLED = "enabled"
    NUM_REQUESTED = "num_requested"
    NUM_COMPLETED = "num_completed"
    START_TIME = "start_time"
    END_TIME = "end_time"
    NUM_SESSION_LOG = "num_session_log"
    NUM_COMMAND = "num_command"
    NUM_ERR_COMMAND = "num_err_command"
    DASHBOARD_VER = "dashboard_ver"
    ORI_DATA_SZ = "ori_data_sz"
    NEW_DATA_SZ = "new_data_sz"
    MODEL_ACCURACY = "model_accuracy"
    MODEL_EPOCH = "model_epoch"
    MODEL_LOSS = "model_loss"


# update job stat interval in seconds
DEFAULT_STAT_UPDATE_INTERVAL = 60


class Recorder:
    """
    Job Management Util.
    Each hitl pipeline run corresponds to a JobManagementUtil instance.
    A stat_update_interal (in seconds) can be specificed for the minimal time to be wait for a repeatly update of the same statistic.
    """

    def __init__(self, stat_update_interval=DEFAULT_STAT_UPDATE_INTERVAL):
        # prepare dict for recording the data
        self._record_dict = {}
        # preoare tmp local file to store job infomation
        time_format = "%Y%m-%d_%H:%M:%S.%f"
        tmp_fname = f"job_management_{datetime.datetime.now().strftime(time_format)}.json"

        folder_path = os.path.join(HITL_TMP_DIR, JOB_MNG_PATH_PREFIX)
        os.makedirs(folder_path, exist_ok=True)
        self._local_path = os.path.join(folder_path, tmp_fname)

        # stat only update if either not set before, or updated for longer than the interval
        self._last_update = datetime.datetime.now()
        self._stat_update_interval = stat_update_interval

    def _set_time(self, time_type, job_type=None):
        """
        record corrent time for the time_type of specified job_type or metadata if job type is None.
        """
        time_now = str(datetime.datetime.now())

        rec_dict = self._record_dict
        tname = time_type._name_

        if job_type is not None:
            self._check_n_init_job(job_type)
            # set job start / end
            jname = job_type._name_
            if tname not in rec_dict[jname]:
                rec_dict[jname][tname] = []
            rec_dict[jname][tname].append(time_now)
        elif tname in rec_dict and rec_dict[tname]:
            # set meta data start / end, can only be set once
            logging.error(
                f"[Job Management Util] Cannot set meta data start/end time twice, ignoring setting {tname}."
            )
        else:
            # set meta data start / end when there is no existing record
            rec_dict[tname] = time_now
        self._save_tmp()

    def _save_tmp(self):
        """
        save json to local temporary folder.
        """
        json.dump(self._record_dict, open(self._local_path, "w"))

    def _check_n_init_job(self, job_type: Job):
        if job_type._name_ not in self._record_dict:
            self._record_dict[job_type._name_] = {}

    def set_meta_start(self):
        """
        record start time of the entire run
        """
        self._set_time(MetaData.START_TIME)

    def set_meta_end(self):
        """
        record end time of the entire run
        """
        self._set_time(MetaData.END_TIME)

    def set_meta_data(self, meta_data: MetaData, val):
        """
        set metadata to the value provided in the input parameter
        """
        self._record_dict[meta_data._name_] = val
        self._save_tmp()

    def set_job_stat(self, job_type: Job, job_stat: JobStat, val, force_update=False):
        """
        set job statistic for the input job type to the value provided in the input parameter.
        you can do a force update regardless of the stat_update_interal with the force_update set to True
        """
        self._check_n_init_job(job_type)

        jname = job_type._name_
        sname = job_stat._name_
        curr_timestamp = datetime.datetime.now()
        since_last_update = curr_timestamp - self._last_update
        since_last_update = since_last_update.total_seconds()

        if (
            sname not in self._record_dict[jname]
            or since_last_update > self._stat_update_interval
            or force_update
        ):
            # update if the not updated before
            # or the duration since last update is larger than the update interval
            # or force update (for job finish status update purpose)
            self._record_dict[jname][sname] = val
            self._last_update = curr_timestamp
            self._save_tmp()

    def set_job_start(self, job_type: Job):
        """
        record the start of a job with the job_type of input
        """
        self._check_n_init_job(job_type)
        self._set_time(JobStat.START_TIME, job_type)

    def set_job_end(self, job_type: Job):
        """
        record the end of a job with the job_type of input
        """
        self._check_n_init_job(job_type)
        self._set_time(JobStat.END_TIME, job_type)

    def save_to_s3(self):
        """
        save local job management record to remote s3 bucket
        """
        batch_id = (
            self._record_dict[MetaData.BATCH_ID._name_]
            if hasattr(MetaData, "BATCH_ID") and MetaData.BATCH_ID._name_ in self._record_dict
            else None
        )
        # check batch_id for saving to s3
        if batch_id is None:
            logging.error(
                f"[Job Management Util] Must have an associated batch to be able to save to s3, please check local path for temporary file job stats record file - {self._local_path}"
            )
        # save to s3
        else:
            remote_file_path = f"{JOB_MNG_PATH_PREFIX}/{batch_id}.json"
            try:
                resp = s3.meta.client.upload_file(
                    self._local_path, S3_BUCKET_NAME, remote_file_path
                )
            except botocore.exceptions.ClientError as e:
                logging.error(
                    f"[Job Management Util] Not able to save file {self._local_path} to s3."
                )
