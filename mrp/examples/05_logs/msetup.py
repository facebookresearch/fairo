import mrp

mrp.process(
    name="alice",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "alice.py"],
    ),
)

mrp.process(
    name="bob",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "bob.py"],
    ),
)

mrp.process(
    name="log",
    runtime=mrp.Docker(
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

if __name__ == "__main__":
    mrp.main()
