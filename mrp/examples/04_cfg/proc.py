import a0
import time

prefix = a0.cfg(a0.env.topic(), "/prefix", str)

i = 0
while True:
    a0.update_configs()
    print(f"{prefix}{i=}")
    i += 1
    time.sleep(1)
