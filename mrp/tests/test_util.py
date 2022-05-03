import os
import threading
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


@pytest.mark.parametrize("logger_type", [util.stdout_logger, util.stderr_logger])
def test_logger(logger_type):
    test_topic = "TEST"
    os.environ["A0_TOPIC"] = test_topic

    # Set up listener
    buffer = {}
    cv = threading.Condition()

    def listener_callback(pkt):
        with cv:
            buffer.update(dict(pkt.headers))
            cv.notify()

    listener = a0.LogListener(test_topic, listener_callback)
    log = logger_type()
    log("test log msg")

    # Test packet contents to ensure content-type is included
    with cv:
        cv.wait_for(lambda: buffer, timeout=1.0)
    assert "content-type" in buffer
    assert buffer["content-type"] == "text/plain"
