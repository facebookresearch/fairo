import a0
import time

i = 0
p = a0.Publisher("some/topic")

while True:
    p.pub(f"{i=}")
    i += 1
    time.sleep(1)
