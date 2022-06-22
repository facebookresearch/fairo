import logging

import types
import threading
import a0
import time

import torch

log = logging.getLogger(__name__)


def compute_quat_dist(a, b):
    return torch.acos((2 * (a * b).sum() ** 2 - 1).clip(-1, 1))


def _get_heartbeat_topic(topic_key):
    return f"{topic_key}/heartbeat"


def start_a0_server_heartbeat(topic_key):
    topic_heartbeat_key = _get_heartbeat_topic(topic_key)
    log.info(f"Starting heartbeat on topic {topic_heartbeat_key}")
    publisher = a0.Publisher(topic_heartbeat_key)

    def beat():
        while True:
            publisher.pub("ready")
            time.sleep(1)

    threading.Thread(target=beat, daemon=True).start()


def wait_until_a0_server_ready(topic_key):
    topic_heartbeat_key = _get_heartbeat_topic(topic_key)
    log.info(f"Waiting for a0 server on topic {topic_heartbeat_key}")

    ns = types.SimpleNamespace(
        cv=threading.Condition(),
        ready=False,
    )

    def callback(pkt):
        with ns.cv:
            ns.ready = True
            ns.cv.notify()

    sub = a0.Subscriber(topic_heartbeat_key, callback)
    with ns.cv:
        ns.cv.wait_for(lambda: ns.ready)
