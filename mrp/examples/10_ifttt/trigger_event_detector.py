import a0
import time

EVENT_TOPIC = a0.cfg(a0.env.topic(), "/event_topic", str)
print(f"{EVENT_TOPIC} initialized")

i = 0
p = a0.Publisher(f"{EVENT_TOPIC}")

while True:
    msg = f"data {i}"
    print(msg)
    p.pub(msg)
    i += 1
    time.sleep(10)
