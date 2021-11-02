import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
)

fbrp.process(
    name="log",
    runtime=fbrp.Docker(
        image="ghcr.io/alephzero/log:latest",
        mount=["/tmp/logs:/tmp/logs"],
    ),
    cfg={
        "savepath": "/tmp/logs",
        "rules": [
            {
                "protocol": "pubsub",
                "topic": "some/topic",
                "policies": [{"type": "save_all"}],
            }
        ],
    },
)

fbrp.main()
