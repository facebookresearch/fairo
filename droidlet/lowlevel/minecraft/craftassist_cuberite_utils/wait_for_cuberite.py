"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import socket
import time
from . import ping_cuberite


def wait_for_server(host, port, wait_ms=200, max_tries=200):
    """This function tries max_tries times to connect to the
    host at the port using a socket connection"""
    for i in range(max_tries):
        try:
            s = socket.socket()
            s.connect((host, port))
            s.close()
            return
        except ConnectionError as e:
            time.sleep(wait_ms / 1000.0)

    # Never came up, throw connection error
    waited_s = wait_ms * max_tries / 1000
    raise ConnectionError("Cuberite not up at port {} after {}s".format(port, waited_s))


def wait_for_cuberite(host, port):
    wait_for_server(host, port)
    ping_cuberite.ping(host, port, timeout=10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=25565)
    args = parser.parse_args()

    wait_for_cuberite(args.host, args.port)
