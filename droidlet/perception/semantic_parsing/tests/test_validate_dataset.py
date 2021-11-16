"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import unittest
from droidlet.perception.semantic_parsing.utils.validate_json import JSONValidator

FULL_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../droidlet/artifacts/datasets/full_data/")
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "../../../documents/json_schema/")


class DataValidationTest(unittest.TestCase):
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
