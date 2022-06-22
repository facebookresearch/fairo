"""
Job management utils
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

# for ecr
AWS_ECR_ACCESS_KEY_ID = os.environ["AWS_ECR_ACCESS_KEY_ID"]
AWS_ECR_SECRET_ACCESS_KEY = os.environ["AWS_ECR_SECRET_ACCESS_KEY"]
AWS_ECR_REGION = os.environ["AWS_ECR_REGION"] if os.getenv("AWS_ECR_REGION") else "us-west-1"
AWS_ECR_REGISTRY_ID = "492338101900"
AWS_ECR_REPO_NAME = "craftassist"

# local tmp directory
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

# job management record path prefix
JOB_MNG_PATH_PREFIX = "job_management_records"

ecr = boto3.client(
    "ecr",
    aws_access_key_id=AWS_ECR_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_ECR_SECRET_ACCESS_KEY,
    region_name=AWS_ECR_REGION,
)

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


# statastics that all jobs have
STAT_FOR_ALL = set(
    [
        JobStat.ENABLED,
        JobStat.NUM_REQUESTED,
        JobStat.NUM_COMPLETED,
        JobStat.START_TIME,
        JobStat.END_TIME,
    ]
)

# statatics that are unique for a job (not in the STAT_FOR_ALL set)
STAT_JOB_PAIR = {
    Job.INTERACTION: set(
        [
            JobStat.NUM_SESSION_LOG,
            JobStat.NUM_COMMAND,
            JobStat.NUM_ERR_COMMAND,
            JobStat.DASHBOARD_VER,
        ]
    ),
    Job.RETRAIN: set(
        [
            JobStat.ORI_DATA_SZ,
            JobStat.NEW_DATA_SZ,
            JobStat.MODEL_ACCURACY,
            JobStat.MODEL_EPOCH,
            JobStat.MODEL_LOSS,
        ]
    ),
}


def get_dashboard_version(image_tag: str):
    response = ecr.batch_get_image(
        registryId=AWS_ECR_REGISTRY_ID,
        repositoryName=AWS_ECR_REPO_NAME,
        imageIds=[
            {"imageTag": image_tag},
        ],
    )
    assert len(response["images"]) == 1
    return response["images"][0]["imageId"]["imageDigest"]


def get_s3_link(batch_id: int):
    return f"https://s3.console.aws.amazon.com/s3/buckets/droidlet-hitl?region={AWS_DEFAULT_REGION}&prefix={batch_id}"


class JobManagementUtil:
    def __init__(self):
        # prepare dict for recording the data
        rec_dict = {}

        for meta_data in MetaData:
            rec_dict[meta_data._name_] = None

        for job in Job:
            rec_dict[job._name_] = {}
            for stat in STAT_FOR_ALL:
                rec_dict[job._name_][stat._name_] = None

            if job in STAT_JOB_PAIR.keys():
                for stat in STAT_JOB_PAIR[job]:
                    rec_dict[job._name_][stat._name_] = None

        self._record_dict = rec_dict
        time_format = "%Y%m-%d_%H:%M:%S.%f"
        tmp_fname = f"job_management_{datetime.datetime.now().strftime(time_format)}.json"

        folder_path = os.path.join(HITL_TMP_DIR, "tmp", JOB_MNG_PATH_PREFIX)
        os.makedirs(folder_path, exist_ok=True)
        self._local_path = os.path.join(folder_path, tmp_fname)

    def _set_time(self, time_type, job_type=None):
        time_now = str(datetime.datetime.now())

        rec_dict = self._record_dict
        tname = time_type._name_

        if job_type is not None:
            # set job start / end
            jname = job_type._name_
            if rec_dict[jname][tname] is None:
                rec_dict[jname][tname] = []
            rec_dict[jname][tname].append(time_now)
        elif rec_dict[tname]:
            # set meta data start / end, can only be set once
            logging.error(
                f"[Job Management Util] Cannot set meta data start/end time twice, ignoring setting {tname}."
            )
        else:
            # set meta data start / end when there is no existing record
            rec_dict[tname] = time_now
        self._save_tmp()

    def _save_tmp(self):
        json.dump(self._record_dict, open(self._local_path, "w"))

    def set_meta_start(self):
        self._set_time(MetaData.START_TIME)

    def set_meta_end(self):
        self._set_time(MetaData.END_TIME)

    def set_meta_data(self, meta_data: MetaData, val):
        self._record_dict[meta_data._name_] = val
        self._save_tmp()

    def set_job_stat(self, job_type: Job, job_stat: JobStat, val):
        self._record_dict[job_type._name_][job_stat._name_] = val
        self._save_tmp()

    def set_job_start(self, job_type: Job):
        self._set_time(JobStat.START_TIME, job_type)

    def set_job_end(self, job_type: Job):
        self._set_time(JobStat.END_TIME, job_type)

    def save_to_s3(self):
        batch_id = self._record_dict[MetaData.BATCH_ID]
        # check batch_id for saving to s3
        if batch_id is None:
            logging.error(
                "[Job Management Util] Must have an associated batch to be able to save to s3"
            )
            raise RuntimeError("No associated batch_id set")
        # save to s3
        remote_file_path = f"{JOB_MNG_PATH_PREFIX}/{batch_id}.json"
        try:
            resp = s3.meta.client.upload_file(self._local_path, S3_BUCKET_NAME, remote_file_path)
        except botocore.exceptions.ClientError as e:
            logging.info(f"[Job Management Util] Not able to save file {self._local_path} to s3.")


if __name__ == "__main__":
    sha256 = get_dashboard_version("cw_test1")
    print(sha256)
