"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import unittest

from base_craftassist_test_case import BaseCraftassistTestCase
from validate_json import JSONValidator

FULL_DATA_DIR = os.path.join(os.path.dirname(__file__), "../agent/datasets/full_data/")
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "../../base_agent/documents/json_schema/")


class DataValidationTest(BaseCraftassistTestCase):
    """This class validates parse trees associated with commonly used commands in ground truth and unit tests.

    Note that ground truth files are symlinked to files in the full data directory.
    """

    def setUp(self):
        # accept only array spans
        self.json_validator_array = JSONValidator(SCHEMA_DIR, "array")

    def test_high_pri_commands(self):
        res = self.json_validator_array.validate_data(
            FULL_DATA_DIR + "high_pri_commands.txt", test_mode=True
        )
        self.assertTrue(res)

    def test_short_commands(self):
        res = self.json_validator_array.validate_data(
            FULL_DATA_DIR + "short_commands.txt", test_mode=True
        )
        self.assertTrue(res)

    def test_annotated(self):
        res = self.json_validator_array.validate_data(
            FULL_DATA_DIR + "annotated.txt", test_mode=True
        )
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
