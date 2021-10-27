import a0

import time

i = 0
p = a0.Publisher("camera/image.a0")
while True:
    msg = f"frame {i}"
    print(msg)
    p.pub(msg)
    i += 1
    time.sleep(1)
