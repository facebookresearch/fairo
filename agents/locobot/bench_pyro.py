import Pyro4
import time

import argparse

parser = argparse.ArgumentParser(description="Pass in server device IP")
parser.add_argument(
    "--ip",
    help="Server device IP. Default is 192.168.0.0",
    type=str,
    default="0.0.0.0",
)

args = parser.parse_args()
                
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION=4

transfer = Pyro4.Proxy("PYRONAME:transfer@" + args.ip)

start_time = time.time_ns()
fps_freq = 1 # displays the frame rate every 1 second
counter = 0

while True:
    counter += 1
    iter_time = time.time_ns() - start_time
    
    if float(iter_time) / 1e9 > fps_freq :
        print("FPS: ", round(counter / (float(iter_time) / 1e9), 1), "  ", int(iter_time / 1e6 / counter), "ms")
        counter = 0
        start_time = time.time_ns()
    transfer.transfer()
