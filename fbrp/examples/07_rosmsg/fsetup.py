import fbrp
import glob

genmsg_py_bin = "${CONDA_PREFIX}/lib/genpy/genmsg_py.py"
msg_files = glob.glob("my_msgs/*.msg")

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
        run_command=f"python {genmsg_py_bin} -p my_msgs -o my_msgs/py "
        + " ".join(msg_files)
        + f" ; python {genmsg_py_bin} -p my_msgs -o my_msgs/py --initpy",
    ),
)

fbrp.main()
