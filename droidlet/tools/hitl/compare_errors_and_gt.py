"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import csv
import argparse
import copy
import logging
import os
import re
import shutil
import signal
import subprocess
import time

from typing import List

import boto3
import botocore

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.client(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def main(opts):
    # List all of the log files in S3 bucket for the given batch_id
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{opts.batch_id}/interaction")
    print(response["Contents"][:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str, required=True, hint="batch ID for interaction job")
    parser.add_argument("nsp_data", type=str, required=True, default="nsp_data_v3.txt")
    opts = parser.parse_args()

    main(opts)
