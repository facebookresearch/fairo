import unittest
import os
from unittest.mock import patch
import boto3
from moto import mock_s3
import botocore.session
import json

from droidlet.tools.hitl.dashboard_app.backend.tests.s3_client_stubber import (
    OS_ENV_DICT,
    MockListObjResult,
    S3ClientMock,
    mock_make_api_call,
)

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"

VALID_ID = 222222222222222222
INVALID_ID = 11111111111111111

HITL_TMP_DIR = (
    os.environ["HITL_TMP_DIR"] if os.getenv("HITL_TMP_DIR") else f"{os.path.expanduser('~')}/.hitl"
)


@mock_s3
class TestAWSHelper(unittest.TestCase):
    def setUp(self):
        conn = boto3.resource("s3", region_name="us-east-1")
        # We need to create the bucket since this is all in Moto's 'virtual' AWS account
        conn.create_bucket(Bucket="droidlet-hitl")
        self._info_fname = f"job_management_records/{VALID_ID}.json"
        self._traceback_fname = f"{VALID_ID}/log_traceback.csv"
        some_dict = {"msg": "hello"}
        json_content = json.dumps(some_dict)
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.put_object(Bucket="droidlet-hitl", Key=self._info_fname, Body=json_content)
        s3.put_object(Bucket="droidlet-hitl", Key=self._traceback_fname, Body="1, 2 \n1, 2")

    @patch.dict("os.environ", OS_ENV_DICT)
    def test_get_job_list(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.put_object(Bucket="droidlet-hitl", Key="20220224132033/", Body="1, 2 \n1, 2")
        from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import get_job_list

        res = get_job_list()
        self.assertGreater(len(res), 0)

    @patch.dict("os.environ", OS_ENV_DICT)
    def test_get_traceback_by_id_valid(self):
        from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
            get_traceback_by_id,
        )

        res, _ = get_traceback_by_id(VALID_ID)
        self.assertIsNotNone(res)
        self.assertNotEqual(res, f"cannot find traceback with id {VALID_ID}")

    @patch.dict("os.environ", OS_ENV_DICT)
    def test_get_traceback_by_id_invalid(self):
        from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
            get_traceback_by_id,
        )

        res, _ = get_traceback_by_id(INVALID_ID)
        self.assertEqual(res, f"cannot find traceback with id {INVALID_ID}")

    @patch.dict("os.environ", OS_ENV_DICT)
    def test_get_run_info_by_id_valid(self):
        from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
            get_run_info_by_id,
        )

        res, _ = get_run_info_by_id(VALID_ID)
        self.assertIsNotNone(res)
        self.assertNotEqual(res, f"cannot find run info with id {VALID_ID}")

    @patch.dict("os.environ", OS_ENV_DICT)
    @mock_s3
    def test_get_run_info_by_id_inalid(self):
        from droidlet.tools.hitl.dashboard_app.backend.dashboard_aws_helper import (
            get_run_info_by_id,
        )

        res, _ = get_run_info_by_id(INVALID_ID)
        self.assertEqual(res, f"cannot find run info with id {INVALID_ID}")

    def tearDown(self):
        # s3.Object(S3_BUCKET_NAME, self._info_fname).delete()
        # s3.Object(S3_BUCKET_NAME, self._traceback_fname).delete()

        # # remove from local temp directory as well
        # local_info_fname = os.path.join(HITL_TMP_DIR, self._info_fname)
        # local_traceback_fname = os.path.join(HITL_TMP_DIR, self._traceback_fname)
        # if os.path.exists(local_info_fname):
        #     os.remove(local_info_fname)
        # if os.path.exists(local_traceback_fname):
        #     os.remove(local_traceback_fname)
        pass


if __name__ == "__main__":
    unittest.main()
