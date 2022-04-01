"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import Pyro4
import time
import os

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")


class RobotMover:
    def __init__(self, ip=None):
        # self.bot = Pyro4.Proxy("PYRONAME:remoterobot@192.168.0.124")
        if not ip:
            self.bot = Pyro4.Proxy("PYRONAME:remoterobot@192.168.1.48")
        else:
            self.bot = Pyro4.Proxy("PYRONAME:remoterobot@" + ip)

    def get_rgb_depth(self):
        print("getting rgb")
        rgb = self.bot.get_rgb()
        depth = self.bot.get_depth()
        return rgb, depth


if __name__ == "__main__":
    IP = "127.0.0.1"
    if os.getenv("ROBOT_IP"):
        IP = os.getenv("ROBOT_IP")
    lc = RobotMover(ip=IP)
    for t in range(5):
        tm = time.time()
        r, d = lc.get_rgb_depth()
        print(time.time() - tm, "seconds")
    print("RGB Shape", r.shape)
    print("Depth Shape", d.shape)
