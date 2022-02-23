import a0
import time

EVENT_TOPIC = a0.cfg(a0.env.topic(), "/event_topic", str)
print(f"{EVENT_TOPIC} initialized")
print(type(EVENT_TOPIC))

i = 0
p = a0.Publisher(f"{EVENT_TOPIC}")
print(f"we have a publisher")

while True:
    print(f"updating config")
    a0.update_configs()
    msg = f"data {i}"
    print(f"stdout {msg}")
    p.pub(msg)
    i += 1
    time.sleep(10)
