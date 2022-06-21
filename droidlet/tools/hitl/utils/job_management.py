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


# statastics that all jobs have
STAT_FOR_ALL = set([JobStat.REQUESTED, JobStat.COMPLETED, JobStat.START_TIME, JobStat.END_TIME])

# statatics that are unique for a job (not in the STAT_FOR_ALL set)
STAT_JOB_PAIR = {
    Job.INTERACTION: set(
        [JobStat.SESSION_LOG, JobStat.COMMAND, JobStat.ERR_COMMAND, JobStat.DASHBOARD_VER]
    ),
    Job.RETRAIN: set([JobStat.ORI_DATA_SZ, JobStat.NEW_DATA_SZ, JobStat.MODEL_ACCURACY]),
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

    def _validate_and_set_time(self, time_type, job_type=None):
        time = str(datetime.datetime.now())

        rec_dict = self._record_dict

        if time_type == MetaData.START_TIME or time_type == MetaData.END_TIME:
            rec_dict[time_type._name_] = time
        elif (
            time_type == JobStat.START_TIME or time_type == JobStat.END_TIME
        ) and job_type is not None:
            rec_dict[job_type._name_][time_type._name_] = time
        else:
            raise RuntimeError(f"Cannot set time for the type {time_type}")

        self._save_tmp()

    def _save_tmp(self):
        json.dump(self._record_dict, open(self._local_path, "w"))

    def set_meta_start(self):
        self.set_meta_time(MetaData.START_TIME)

    def set_meta_end(self):
        self.set_meta_time(MetaData.END_TIME)

    def set_meta_time(self, meta_data: MetaData):
        self._validate_and_set_time(meta_data)

    def set_meta_data(self, meta_data: MetaData, val):
        self._record_dict[meta_data._name_] = val
        self._save_tmp()

    def set_job_stat(self, job_type: Job, job_stat: JobStat, val):
        self._record_dict[job_type._name_][job_stat._name_] = val
        self._save_tmp()

    def set_job_time(self, job_type: Job, job_stat: JobStat):
        self._validate_and_set_time(job_type, job_stat)

    def save_to_s3(self):
        batch_id = self._record_dict[MetaData.BATCH_ID]
        # check batch_id for saving to s3
        if batch_id is None:
            logging.error("Must have an associated batch to be able to save to s3")
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
