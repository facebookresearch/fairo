"""
Copyright (c) Facebook, Inc. and its affiliates.

This file is a helper for dashboard server,
it provides helper method to interact with aws s3 
and preparing proper response for APIs the server provides.
"""

import json
import tarfile
import boto3
import botocore
import os
import re

from droidlet.tools.hitl.dashboard_app.backend.dashboard_model_utils import load_model


PIPELINE_DATASET_MAPPING = {
    "NLU": "nsp_data",
}

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
        return f"cannot find traceback with id {batch_id}", 404
    return _read_file(local_fname), None


def get_run_info_by_id(batch_id: int):
    """
    helper method for preparing get_run_info_by_id api's response
    """
    local_fname = _download_file(f"job_management_records/{batch_id}.json")
    if local_fname is None:
        return f"cannot find run info with id {batch_id}", 404
    f = open(local_fname)
    json_data = json.load(f)
    f.close()
    return json_data, None


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


def get_interaction_session_log_by_id(batch_id: int, session_id: str):
    """
    helper method to reterive session log
    """
    local_fname = _download_file(f"{batch_id}/interaction/{session_id}/logs.tar.gz")
    if local_fname is None:
        return f"cannot find log with batch_id {batch_id}, session_id {session_id}", 404
    folder_path = local_fname[: local_fname.rindex("/")]
    file = tarfile.open(local_fname)
    file.extractall(folder_path)
    file.close()

    # read agent log
    log_fname = f"{folder_path}/agent.log"
    log_file = open(log_fname, "r")
    log = log_file.readlines()
    return log, None


def get_dataset_indices_by_id(batch_id: int):
    """
    For a run specied by the batch id,
    get the newly added dataset data points' start index and end index from meta.txt
    """
    local_fname = _download_file(f"{batch_id}/meta.txt")
    if local_fname is None:
        return f"cannot find meta.txt with id {batch_id}", 404
    meta_txt_content = _read_file(local_fname)
    meta_txt_splitted = meta_txt_content.split("\n")
    return [int(meta_txt_splitted[0]), int(meta_txt_splitted[-1])], None


def get_dataset_version_list_by_pipeline(pipeline: str):
    """
    Get dataset's version list
    """
    # get dataset name prefix and search pattern
    dataset_prefix = PIPELINE_DATASET_MAPPING[pipeline]
    pattern_str = f"{dataset_prefix}_v[0-9]" + "{1,}.txt"
    pattern = re.compile(pattern_str)

    dataset_list = []
    # list object from s3 bucket
    for obj in s3.Bucket(S3_BUCKET_NAME).objects.all():
        if re.match(pattern, obj.key):
            dataset_list.append(obj.key)
    return dataset_list


def get_dataset_by_name(dataset_name: str):
    """
    Reterive a specified dataset's content
    """
    local_fname = _download_file(dataset_name)
    if local_fname is None:
        return f"cannot find {dataset_name}", 404
    return _read_file(local_fname), None


def get_model_by_id(batch_id: int):
    """
    Download best model from aws, return the model if the model file exists
    """
    local_fname = _download_file(f"{batch_id}/best_model/best_model.pth")
    if local_fname is None:
        return f"cannot find best_model file related to {batch_id}", 404
    else:
        return load_model(local_fname), None
