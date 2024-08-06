import mrp

genmsg_py = [
    "python",
    mrp.NoEscape("${CONDA_PREFIX}/lib/genpy/genmsg_py.py"),
    "-p",
    "my_msgs",
    "-o",
    "my_msgs/py",
]

mrp.process(
    name="genmsgs",
    runtime=mrp.Conda(
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

if __name__ == "__main__":
    mrp.main()
