import a0
import signal
import requests

EVENT_NAME = a0.cfg(a0.env.topic(), "/ifttt/event_name", str)
KEY = a0.cfg(a0.env.topic(), "/ifttt/key", str)
EVENT_TOPIC = a0.cfg(a0.env.topic(), "/event_topic", str)
a0.update_configs()


def call_robo_crit_event(pkt):
    print(f"Pinging IFTTT Event {EVENT_NAME}")
    requests.post(f"https://maker.ifttt.com/trigger/{EVENT_NAME}/with/key/{KEY}")


s = a0.Subscriber(f"{EVENT_TOPIC}", lambda pkt: call_robo_crit_event(pkt))

# a0.update_configs()
signal.pause()
