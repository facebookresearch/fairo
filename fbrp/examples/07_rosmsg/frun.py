import fbrp

fbrp.process(
    name="proc",
    runtime=fbrp.Conda(
        channels=[
            "conda-forge",
            "robostack",
        ],
        dependencies=[
            "python>=3.7",
            "ros-common-msgs",
        ],
        run_command=["python", "proc.py"],
    ),
)

fbrp.main()
