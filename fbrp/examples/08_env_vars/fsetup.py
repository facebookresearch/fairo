import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
    ),
    env={
        "my_path" : "/tmp/",
        "my_time_out" : 2,
    },
)

fbrp.main()
