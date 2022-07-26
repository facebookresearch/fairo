import mrp

mrp.process(
    name="proc",
    runtime=mrp.Conda(
        yaml="env.yml",
        run_command=["python3", "proc.py"],
    ),
    cfg={
        "prefix": "In the beginning... ",
    },
)

mrp.process(
    name="api",
    runtime=mrp.Docker(image="ghcr.io/alephzero/api:latest"),
)

if __name__ == "__main__":
    mrp.main()
