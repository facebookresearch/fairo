import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7", {"pip": ["alephzero"]}],
        run_command=["python3", "proc.py"],
    ),
    cfg={
        "prefix": "In the beginning... ",
    },
)

fbrp.process(
    name="api",
    runtime=fbrp.Docker(image="ghcr.io/alephzero/api:latest"),
)

fbrp.main()
