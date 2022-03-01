"""Dummy data generator."""

import a0
import time

p = a0.Publisher("data")

i = 0
while True:
    p.pub(f"msg {i}")
    i += 1
    time.sleep(1)
