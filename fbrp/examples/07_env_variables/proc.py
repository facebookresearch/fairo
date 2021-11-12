import os
import time

i = 0
assert os.getenv("my_path") is "/highway"
assert os.getenv("my_timeout") == 1
assert os.getenv("policy_path") is "/tmp/policies"
while True:
    print(f"{i=}")
    i += 1
    time.sleep(1)
