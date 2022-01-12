import fbrp

genmsg_py = [
    "python",
    fbrp.NoEscape("${CONDA_PREFIX}/lib/genpy/genmsg_py.py"),
    "-p",
    "my_msgs",
    "-o",
    "my_msgs/py",
]

fbrp.process(
    name="genmsgs",
    runtime=fbrp.Conda(
        channels=[
            "conda-forge",
            "robostack",
        ],
        dependencies=[
            "ros-noetic-genpy",
        ],
        setup_commands=[
            genmsg_py + ["my_msgs/PsUtil.msg"],
            genmsg_py + ["--initpy"],
        ],
        run_command=[],
    ),
)

fbrp.main()
