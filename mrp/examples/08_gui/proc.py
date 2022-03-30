import os
import time

i = 0
while True:
    print(f"{i=}")
    print(os.environ["my_conda_path"])
    print(os.environ["my_conda_time_out"])
    i += 1
    time.sleep(1)
