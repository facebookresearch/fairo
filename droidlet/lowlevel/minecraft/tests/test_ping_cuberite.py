"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import socket
import unittest


class CuberiteBasicTest(unittest.TestCase):

    def test_something(self):
        # Raw bytes of handshake and ping packets, see http://wiki.vg/Protocol#Ping
        handshake_ping = b"\x10\x00\xd4\x02\tlocalhost\x63\xdd\x01\t\x01\x00\x00\x00\x00\x00\x00\x00*"

        # Raw bytes of clientbound pong packet, see http://wiki.vg/Protocol#Pong
        response_pong = b"\x09\x01\x00\x00\x00\x00\x00\x00\x00\x2a"

        """Ping craftassist_cuberite_utils using a socket connection"""
        print("Sending ping to {}:{}".format("localhost", 25565))
        s = socket.socket()
        s.connect(("localhost", 25565))
        s.settimeout(10)
        s.send(handshake_ping)
        response = s.recv(len(response_pong))
        print("Received pong!")
        assert response == response_pong, "Bad pong: {}".format(response)


if __name__ == '__main__':
    unittest.main()
