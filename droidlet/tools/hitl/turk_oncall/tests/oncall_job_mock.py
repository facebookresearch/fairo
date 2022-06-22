"""
Copyright (c) Facebook, Inc. and its affiliates.

This file incldue a mock class of OnCallJob, 
which add a stat to the specified batch_id's corresponding batch location. 
"""
import boto3
import botocore
import logging
import os

from droidlet.tools.hitl.data_generator import DataGenerator

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


class MockOnCallJob(DataGenerator):
    # for test purpose
    def __init__(self, batch_id: int, timeout: float = -1) -> None:
        super().__init__(timeout)
        self._batch_id = batch_id

    def run(self) -> None:
        batch_id = self._batch_id

        # Add stat file
        stat_fname = f"{batch_id}.stat"

        obj = s3.Object(S3_BUCKET_NAME, f"{batch_id}/{stat_fname}")
        result = obj.put(Body="ready").get("ResponseMetadata")
        if result.get("HTTPStatusCode") == 200:
            self.set_finished()
        else:
            logging.info(f"[Oncall Job Mock] {batch_id}.stat not updated")
