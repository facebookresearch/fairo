"""
Copyright (c) Facebook, Inc. and its affiliates.

This file is a helper for dashboard server,
it provides helper method to interact with aws s3 
and preparing proper response for APIs the server provides.
"""

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


def _download_file(fname: str):
    """
    download file from s3 if it does not exists in local tmp storage
    """
    # check if exists on local tmp directory
    local_file_name = os.path.join(HITL_TMP_DIR, fname)

    if not os.path.exists(local_file_name):
        # reterive from s3
        local_folder_name = local_file_name[: local_file_name.rindex("/")]
        os.makedirs(local_folder_name, exist_ok=True)
        try:
            s3.meta.client.download_file(S3_BUCKET_NAME, fname, local_file_name)
        except botocore.exceptions.ClientError as e:
            print(f"file not exists {fname}")

    return local_file_name if os.path.exists(local_file_name) else None


def _read_file(fname: str):
    """
    read file into a string
    """
    f = open(fname, "r")
    content = f.read()
    f.close()
    return content


def get_job_list():
    """
    helper method for preparing get_job_list api's response
    """
    job_list = []
    # list object from s3 bucket
    res = s3.meta.client.get_paginator("list_objects").paginate(
        Bucket=S3_BUCKET_NAME, Delimiter="/"
    )
    # pattern of YYYYMMDDHHMMSS (batch id pattern)
    pattern = r"([0-9]{4})(0[1-9]|1[0-2])(0[1-9]|[1-2][0-9]|3[0-1])(2[0-3]|[01][0-9])([0-5][0-9])([0-5][0-9])"

    for prefix in res.search("CommonPrefixes"):
        if re.match(pattern, prefix.get("Prefix")):
            job_list.append(int(prefix.get("Prefix")[:-1]))
    return job_list


def get_traceback_by_id(batch_id: int):
    """
    helper method for preparing get_traceback_by_id api's response
    """
    local_fname = _download_file(f"{batch_id}/log_traceback.csv")
    if local_fname is None:
        return f"cannot find traceback with id {batch_id}"
    return _read_file(local_fname)


def get_run_info_by_id(batch_id: int):
    """
    helper method for preparing get_run_info_by_id api's response
    """
    local_fname = _download_file(f"job_management_records/{batch_id}.json")
    if local_fname is None:
        return f"cannot find run info with id {batch_id}"
    f = open(local_fname)
    json_data = json.load(f)
    f.close()
    return json_data


def get_interaction_sessions_by_id(batch_id: int):
    session_list = []
    s3_bucket = s3.Bucket(S3_BUCKET_NAME)
    prefix = f"{batch_id}/interaction/"

    for obj in s3_bucket.objects.filter(Prefix=prefix):
        session_name = obj.key
        left_idx = session_name.index(prefix) + len(prefix)
        right_idx = session_name.index("/logs.tar.gz")
        session_list.append(session_name[left_idx:right_idx])
    return session_list
