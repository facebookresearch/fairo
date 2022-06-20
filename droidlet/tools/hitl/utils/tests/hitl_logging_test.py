"""
Copyright (c) Facebook, Inc. and its affiliates.

Unit test for hitl_logging.py
"""

from pathlib import Path
import shutil
from droidlet.tools.hitl.utils.hitl_logging import HitlLogging
from droidlet.tools.hitl.utils.hitl_utils import generate_batch_id
import unittest
import os

# Generate a batch id
batch_id = generate_batch_id()
msg = "I am a message"


class TestLoggerClass1:
    def __init__(self) -> None:
        pass

    def call_logging(self):
        # Log a message
        hl = HitlLogging(batch_id)
        hl_logger = hl.get_logger()
        hl_logger.error(msg)
        return hl


class TestLoggerClass2:
    def __init__(self) -> None:
        pass

    def call_logging(self):
        # Log a message
        hl = HitlLogging(batch_id)
        hl_logger = hl.get_logger()
        hl_logger.warning(msg)
        return hl


class TestHitlLogging(unittest.TestCase):
    def setUp(self):
        test_cls = []
        test_cls.append(TestLoggerClass1())
        test_cls.append(TestLoggerClass2())
        test_cls.append(TestLoggerClass2())

        log_files = []

        for cl in test_cls:
            hl = cl.call_logging()
            log_files.append(hl.get_log_file())
            hl.shutdown()

        self._log_files = log_files

    def test_hitl_logging_3instances(self):
        log_files = self._log_files

        # should generate 3 log files
        self.assertEqual(len(log_files), 3)

        # check log file content
        for log_fname in log_files:
            self.assertTrue(os.path.exists(log_fname))
            log_f = open(log_fname, "r")
            for line in log_f:
                self.assertTrue(msg in line)
            log_f.close()

    def tearDown(self):
        log_fname = self._log_files[0]
        path = os.path.dirname(log_fname)
        path = Path(path).parent.absolute()
        shutil.rmtree(path=path)


if __name__ == "__main__":
    unittest.main()
