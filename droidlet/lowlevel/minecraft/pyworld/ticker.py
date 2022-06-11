"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import socketio
import time


class Ticker:
    def __init__(
        self,
        tick_rate,
        step_rate,
        ip,
        port,
    ):
        self.tick_rate = tick_rate
        self.step_rate = step_rate
        self.sio = socketio.Client()
        self.ip = ip
        self.port = port
        self.time = time.time()

    def start(self):
        """
        step the world every step_rate seconds
        sleep for tick_rate seconds between polls.
        """

        time.sleep(5)  # give world enough time to initialize
        self.sio.connect("http://{}:{}".format(self.ip, self.port))

        self.time = time.time()
        while True:
            t = time.time()
            if t - self.time > self.step_rate:
                self.sio.emit("step_world")
                self.time = t
            time.sleep(self.tick_rate)


if __name__ == "__main__":
    ticker = Ticker(tick_rate=0.01, step_rate=0.2, ip="localhost", port=6002)
    ticker.start()
