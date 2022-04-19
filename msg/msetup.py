import mrp


ros_msg_packages = [
    "ros-noetic-std-msgs",
    "ros-noetic-common-msgs",
]

mrp.process(
    name="gen",
    runtime=mrp.Conda(
        channels=["conda-forge", "robostack"],
        dependencies=[
            "python=3.8",
            "capnproto",
        ]
        + ros_msg_packages,
        run_command=["python", "scripts/gen.py"],
    ),
)

mrp.main()
