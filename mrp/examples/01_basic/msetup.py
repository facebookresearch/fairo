import mrp

mrp.process(
    name="proc",
    runtime=mrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
)

mrp.main()
