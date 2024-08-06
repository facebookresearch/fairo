import mrp

IFTTT_KEY = "YOUR_KEY_GOES_HERE"
TRIGGER_EVENT_NAME = "YOUR_EVENT_NAME_GOES_HERE"
TRIGGER_EVENT_TOPIC = "YOUR_TOPIC_NAME_GOES_HERE"

mrp.process(
    name="trigger_event_detector",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "trigger_event_detector.py"],
    ),
    cfg={
        "event_topic": TRIGGER_EVENT_TOPIC,
    },
)

mrp.process(
    name="ifttt_webhook",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "ifttt.py"],
    ),
    cfg={
        "ifttt": {"key": IFTTT_KEY, "event_name": TRIGGER_EVENT_NAME},
        "event_topic": TRIGGER_EVENT_TOPIC,
    },
)

if __name__ == "__main__":
    mrp.main()
