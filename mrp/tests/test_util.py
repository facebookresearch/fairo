import os
import time
import pytest

import a0

from mrp import util


def test_empty():
    assert util.shell_join([]) == ""


def test_basic():
    assert util.shell_join(["a", "b", "c"]) == "a b c"


def test_escape():
    assert util.shell_join(["a", ";b", "c"]) == "a ';b' c"


def test_noescape():
    assert util.shell_join(["a", util.NoEscape(";b"), "c"]) == "a ;b c"


class TestLogger:
    def _listener_callback(self, pkt):
        self.buffer = {h[0]: h[1] for h in pkt.headers}

    @pytest.mark.parametrize("logger_type", [util.stdout_logger, util.stderr_logger])
    def test_logger(self, log_func):
        self.buffer = None
        test_topic = "TEST"
        os.environ["A0_TOPIC"] = test_topic

        # Set up listener
        listener = a0.LogListener(a0.LogTopic(test_topic), self._listener_callback)

        log = logger_type()
        log("test log msg")
        time.sleep(0.1)  # wait for log to publish

        # Test packet contents to ensure content-type is included
        assert self.buffer is not None
        assert "content-type" in self.buffer
        assert self.buffer["content-type"] == "text/plain"
