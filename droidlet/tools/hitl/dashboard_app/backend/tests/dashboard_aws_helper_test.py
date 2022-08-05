import unittest
import os
import boto3
import json

from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
    get_job_list,
    get_run_info_by_id,
    get_traceback_by_id,
)

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

VALID_ID = 222222222222222222
INVALID_ID = 11111111111111111

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)

s3 = boto3.resource(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


class TestAWSHelper(unittest.TestCase):
    def setUp(self):
        self._info_fname = f"job_management_records/{VALID_ID}.json"
        self._traceback_fname = f"{VALID_ID}/log_traceback.csv"
        some_dict = {"msg": "hello"}
        json_content = json.dumps(some_dict)

        s3.Object(S3_BUCKET_NAME, self._info_fname).put(Body=json_content)
        s3.Object(S3_BUCKET_NAME, self._traceback_fname).put(Body="1, 2 \n1, 2")

    def test_get_job_list(self):
        res = get_job_list()
        self.assertGreater(len(res), 0)

    def test_get_traceback_by_id_valid(self):
        res, _ = get_traceback_by_id(VALID_ID)
        self.assertIsNotNone(res)
        self.assertNotEqual(res, f"cannot find traceback with id {VALID_ID}")

    def test_get_traceback_by_id_invalid(self):
        res, _ = get_traceback_by_id(INVALID_ID)
        self.assertEqual(res, f"cannot find traceback with id {INVALID_ID}")

    def test_get_run_info_by_id_valid(self):
        res, _ = get_run_info_by_id(VALID_ID)
        self.assertIsNotNone(res)
        self.assertNotEqual(res, f"cannot find run info with id {VALID_ID}")

    def test_get_run_info_by_id_inalid(self):
        res, _ = get_run_info_by_id(INVALID_ID)
        self.assertEqual(res, f"cannot find run info with id {INVALID_ID}")

    def tearDown(self):
        s3.Object(S3_BUCKET_NAME, self._info_fname).delete()
        s3.Object(S3_BUCKET_NAME, self._traceback_fname).delete()

        # remove from local temp directory as well
        local_info_fname = os.path.join(HITL_TMP_DIR, self._info_fname)
        local_traceback_fname = os.path.join(HITL_TMP_DIR, self._traceback_fname)
        if os.path.exists(local_info_fname):
            os.remove(local_info_fname)
        if os.path.exists(local_traceback_fname):
            os.remove(local_traceback_fname)


if __name__ == "__main__":
    unittest.main()
