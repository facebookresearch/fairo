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
    name="api",
    runtime=mrp.Docker(image="ghcr.io/alephzero/api:latest"),
)

if __name__ == "__main__":
    mrp.main()
