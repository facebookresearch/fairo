import fbrp

fbrp.process(
    name="alice",
    runtime=fbrp.Conda(
        yaml="env.yml",
        run_command=["python3", "alice.py"],
    ),
)

fbrp.process(
    name="bob",
    runtime=fbrp.Conda(
        yaml="env.yml",
        run_command=["python3", "bob.py"],
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
                "topic": "from/*",
                "policies": [{"type": "save_all"}],
            }
        ],
    },
)

fbrp.main()
