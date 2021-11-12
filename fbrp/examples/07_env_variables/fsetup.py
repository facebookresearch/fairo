import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        dependencies=["python>=3.7"],
        run_command=["python3", "proc.py"],
        env_variables = {
            "my_path" : "/highway",
            "my_timeout" : 1,
            "policy_path": "/tmp/policies",
        },
    ),
)

fbrp.main()
