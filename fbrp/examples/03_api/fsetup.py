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
    name="api",
    runtime=fbrp.Docker(image="ghcr.io/alephzero/api:latest"),
)

fbrp.main()
