import mrp

mrp.process(
    name="proc",
    runtime=mrp.Conda(
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

mrp.main()
