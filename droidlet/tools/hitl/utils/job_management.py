"""
Job management utils
"""
import logging
from math import isnan
import boto3
import datetime
import pandas as pd
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
    ID = "id"
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
    REQUESTED = "#requested"
    COMPLETED = "#completed"
    START_TIME = "start_time"
    END_TIME = "end_time"
    SESSION_LOG = "#session_log"
    COMMAND = "#command"
    ERR_COMMAND = "#err_command"
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


def get_job_stat_col(job: Job, job_stat: JobStat):
    """
    Gets Job statstic column name if the this job_stat is allowed for the input job
    """
    assert job_stat in STAT_FOR_ALL or job_stat in STAT_JOB_PAIR[job]

    return f"{job.name}.{job_stat.name}"


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


# Pepare record columns
rec_cols = [md.name for md in MetaData]

for job in Job:
    for stat in STAT_FOR_ALL:
        rec_cols.append(get_job_stat_col(job, stat))
    if job in STAT_JOB_PAIR.keys():
        for stat in STAT_JOB_PAIR[job]:
            rec_cols.append(get_job_stat_col(job, stat))


class JobManagementUtil:
    def __init__(self):
        df = pd.DataFrame(columns=rec_cols, index=[0])
        self._batch_id = None
        self._rec_df = df
        self._local_path = os.path.join(
            HITL_TMP_DIR, "tmp", f"job_management_{datetime.datetime.now()}.csv"
        )

    def _validate_and_set_time(self, time_type, job_type=None):
        time = datetime.datetime.now()

        df = self._rec_df

        # Check type and get corresponding column name
        if time_type == MetaData.START_TIME or time_type == MetaData.END_TIME:
            start_col = MetaData.START_TIME.name
            col_to_set = time_type.name
        elif (
            time_type == JobStat.START_TIME or time_type == JobStat.END_TIME
        ) and job_type is not None:
            start_col = get_job_stat_col(job_type, JobStat.START_TIME)
            col_to_set = get_job_stat_col(job_type, time_type)
        else:
            raise TypeError(f"Cannot set time for the type {time_type}")

        # Validate has start time for recording end time
        if time_type == MetaData.END_TIME or time_type == JobStat.END_TIME:
            # start time need to be set
            assert not isnan(df.at[0, start_col])

        # Validate not set before
        assert isnan(df.at[0, col_to_set])

        # Set time
        df.at[0, col_to_set] = time
        self._save_tmp()

    def _save_tmp(self):
        self._rec_df.to_csv(self._local_path)

    def set_meta_start(self):
        self.set_meta_time(MetaData.START_TIME)

    def set_meta_end(self):
        self.set_meta_time(MetaData.END_TIME)

    def set_meta_time(self, meta_data: MetaData):
        self.validate_and_set_time(meta_data)

    def set_meta_data(self, meta_data: MetaData, val):
        self._rec_df.at[0, meta_data.name] = val
        self._save_tmp()

    def set_job_stat(self, job_type: Job, job_stat: JobStat, val):
        self._rec_df.at[0, get_job_stat_col(job_type, job_stat)] = val
        self._save_tmp()

    def set_job_stat_relative(self, job_type: Job, job_stat: JobStat, relative_val):
        self._rec_df.at[0, get_job_stat_col(job_type, job_stat)] += relative_val
        self._save_tmp()

    def set_job_time(self, job_type: Job, job_stat: JobStat):
        self.validate_and_set_time(job_type, job_stat)

    def save_to_s3(self):
        # check __batch_id for saving to s3
        if self._batch_id is None:
            logging.error("Must have an associated batch to be able to save to s3")
            raise TypeError("No associated batch_id set")
        # save to s3


if __name__ == "__main__":
    sha256 = get_dashboard_version("cw_test1")
    print(sha256)
