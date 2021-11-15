import os
import time

assert os.environ["my_path"] is "/tmp/"
assert os.environ["my_time_out"] == 2

i = 0
while True:
    print(f"{i=}")
    i += 1
    time.sleep(1)
