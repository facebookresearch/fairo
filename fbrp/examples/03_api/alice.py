import a0
import sys
import time

i = 0
p = a0.Publisher("some/topic")
l = a0.Logger(a0.env.topic())

while True:
    msg = f"data {i}"
    print(f"stdout {msg}")
    print(f"stderr {msg}", file=sys.stderr)
    l.crit(f"crit {msg}")
    p.pub(msg)
    i += 1
    time.sleep(1)
