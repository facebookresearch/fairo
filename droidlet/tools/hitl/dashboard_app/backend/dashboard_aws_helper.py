import boto3
import os
import re

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


def get_job_info(job_id: int):
    pass
