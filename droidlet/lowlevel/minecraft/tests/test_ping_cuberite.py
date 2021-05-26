"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import socket
import unittest
import subprocess
from droidlet.lowlevel.minecraft.cuberite_process import CuberiteProcess, create_workdir

class CuberiteBasicTest(unittest.TestCase):
    def test_create_workdir(self):
        # diverse_world
        plugins = ["debug", "chatlog", "point_blocks"]
        create_workdir(config_name="flat_world", seed=0, game_mode="creative", port=25565, plugins=plugins, place_blocks_yzx=None, workdir_root=None)

    # def test_something(self):
    #     repo_home = os.path.dirname(os.path.realpath(__file__))
    #
    #     print("Cuberite workdir: {}".format(self.workdir))
    #     popen = [repo_home + "/server/cuberite/Server/Cuberite"]
    #     p = subprocess.Popen(popen, cwd=self.workdir, stdin=subprocess.PIPE)
    #     # Raw bytes of handshake and ping packets, see http://wiki.vg/Protocol#Ping
    #     handshake_ping = b"\x10\x00\xd4\x02\tlocalhost\x63\xdd\x01\t\x01\x00\x00\x00\x00\x00\x00\x00*"
    #
    #     # Raw bytes of clientbound pong packet, see http://wiki.vg/Protocol#Pong
    #     response_pong = b"\x09\x01\x00\x00\x00\x00\x00\x00\x00\x2a"
    #
    #     """Ping craftassist_cuberite_utils using a socket connection"""
    #     print("Sending ping to {}:{}".format("localhost", 25565))
    #     s = socket.socket()
    #     s.connect(("localhost", 25565))
    #     s.settimeout(10)
    #     s.send(handshake_ping)
    #     response = s.recv(len(response_pong))
    #     print("Received pong!")
    #     assert response == response_pong, "Bad pong: {}".format(response)


if __name__ == '__main__':
    unittest.main()
