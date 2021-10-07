import a0
import time

i = 0
p = a0.Publisher("imu/image")
while True:
    msg = f"data {i}"
    print(msg)
    p.pub(msg)
    i += 1
    time.sleep(1)
