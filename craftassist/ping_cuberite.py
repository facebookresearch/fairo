"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import socket

# Raw bytes of handshake and ping packets, see http://wiki.vg/Protocol#Ping
HANDSHAKE_PING = b"\x10\x00\xd4\x02\tlocalhost\x63\xdd\x01\t\x01\x00\x00\x00\x00\x00\x00\x00*"

# Raw bytes of clientbound pong packet, see http://wiki.vg/Protocol#Pong
PONG = b"\x09\x01\x00\x00\x00\x00\x00\x00\x00\x2a"


def ping(host, port, timeout=10):
    """Ping cuberite using a socket connection"""
    s = socket.socket()
    s.connect((host, port))
    s.settimeout(timeout)
    s.send(HANDSHAKE_PING)
    response = s.recv(len(PONG))
    assert response == PONG, "Bad pong: {}".format(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=25565)
    args = parser.parse_args()

    print("Sending ping to {}:{}".format(args.host, args.port))
    ping(args.host, args.port)
    print("Received pong!")
