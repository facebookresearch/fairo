import fbrp

IFTTT_KEY = "YOUR_KEY_GOES_HERE"
TRIGGER_EVENT_NAME = "YOUR_EVENT_NAME_GOES_HERE"
TRIGGER_EVENT_TOPIC = "YOUR_TOPIC_NAME_GOES_HERE"

fbrp.process(
    name="trigger_event_detector",
    runtime=fbrp.Conda(
        yaml="env.yml",
        run_command=["python3", "trigger_event_detector.py"],
    ),
    cfg={
        "event_topic": TRIGGER_EVENT_TOPIC,
    },
)

fbrp.process(
    name="ifttt_webhook",
    runtime=fbrp.Conda(
        yaml="env.yml",
        run_command=["python3", "ifttt.py"],
    ),
    cfg={
        "ifttt": {"key": IFTTT_KEY, "event_name": TRIGGER_EVENT_NAME},
        "event_topic": TRIGGER_EVENT_TOPIC,
    },
)

fbrp.main()
