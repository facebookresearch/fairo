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
    def test_hitl_logging_3instances(self):
        test_cl1 = TestLoggerClass1()
        test_cl2 = TestLoggerClass2()
        test_cl2_instance2 = TestLoggerClass2()

        hl1 = test_cl1.call_logging()
        hl2 = test_cl2.call_logging()
        hl3 = test_cl2_instance2.call_logging()

        # should generate 3 log files
        log_files = []
        log_files.append(hl1.get_log_file())
        log_files.append(hl2.get_log_file())
        log_files.append(hl3.get_log_file())

        for log_fname in log_files:
            self.assertTrue(os.path.exists(log_fname))
            log_f = open(log_fname, "r")
            for line in log_f:
                self.assertTrue(msg in line)
            log_f.close()


if __name__ == "__main__":
    unittest.main()
