import json
import boto3
import botocore
import os
import re

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

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


def _dowload_file(fname: str):
    # check if exists on local tmp directory
    local_file_name = os.path.join(HITL_TMP_DIR, fname)

    if not os.path.exists(local_file_name):
        # reterive from s3
        local_folder_name = local_file_name[: local_file_name.rindex("/")]
        os.makedirs(local_folder_name, exist_ok=True)
        try:
            s3.meta.client.download_file(S3_BUCKET_NAME, fname, local_file_name)
        except botocore.exceptions.ClientError as e:
            print("file not exists")

    return local_file_name if os.path.exists(local_file_name) else None


def _read_file(fname: str):
    f = open(fname, "r")
    content = f.read()
    f.close()
    return content


def get_job_list():
    job_list = []
    res = bucket.meta.client.get_paginator("list_objects").paginate(
        Bucket=S3_BUCKET_NAME, Delimiter="/"
    )
    pattern = r"([0-9]{4})(0[1-9]|1[0-2])(0[1-9]|[1-2][0-9]|3[0-1])(2[0-3]|[01][0-9])([0-5][0-9])([0-5][0-9])"

    for prefix in res.search("CommonPrefixes"):
        if re.match(pattern, prefix.get("Prefix")):
            job_list.append(int(prefix.get("Prefix")[:-1]))
    return job_list


def get_traceback_by_id(job_id: int):
    local_fname = _dowload_file(f"{job_id}/log_traceback.csv")
    if local_fname is None:
        return f"cannot find traceback with id {job_id}"
    return _read_file(local_fname)


def get_run_info_by_id(job_id: int):
    local_fname = _dowload_file(f"job_management_records/{job_id}.json")
    if local_fname is None:
        return f"cannot find run info with id {job_id}"
    f = open(local_fname)
    json_data = json.load(f)
    f.close()
    return json_data
