"""
Copyright (c) Facebook, Inc. and its affiliates.

The oncall_bug_report_offline.py is a script for crawling past HiTL batches 
within the range of [start_batch_id, end_batch_id], and extract bug report offline.
"""
import argparse
import boto3
from datetime import datetime
import os

from droidlet.tools.hitl.utils.hitl_utils import generate_batch_id
from droidlet.tools.hitl.turk_oncall.oncall_bug_report import process_past_logs


S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

s3_client = boto3.client(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
TIME_FORMAT = "%Y%m%d%H%M%S"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--start_batch_id",
        type=int,
        required=True,
        help="Process region start, inclusive",
    )

    parser.add_argument(
        "-e",
        "--end_batch_id",
        type=int,
        required=False,
        help="Process region end (optional, default to now), inclusive",
    )
    opts = parser.parse_args()

    start_batch_id = opts.start_batch_id
    end_batch_id = opts.end_batch_id if opts.end_batch_id else generate_batch_id()

    start_datetime = datetime.strptime(str(start_batch_id), TIME_FORMAT)
    end_datetime = datetime.strptime(str(end_batch_id), TIME_FORMAT)

    paginator = s3_client.get_paginator("list_objects")
    result = paginator.paginate(Bucket=S3_BUCKET_NAME, Delimiter="/")

    # crawl batch id from s3 bucket
    batch_ids = []
    for prefix in result.search("CommonPrefixes"):
        name = prefix.get("Prefix")[:-1]

        batch_id = int(name) if name.isdigit() else None

        if batch_id:
            try:
                batch_datetime = datetime.strptime(str(batch_id), TIME_FORMAT)
                if batch_datetime >= start_datetime and batch_datetime <= end_datetime:
                    batch_ids.append(batch_id)
            except ValueError:
                pass

    # process logs
    process_past_logs(batch_ids)
