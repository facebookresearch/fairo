import mrp
import os
import pathlib
import socket

# TODO(lshamis): Find a reliable way to detect IP.
# The get_ip method below will return the same result as
# default_ip in the launch.sh file, but it may not be the
# tailscale ip.
#
# def get_ip():
#   s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#   # Doesn't need to actually connect.
#   s.connect(("8.8.8.8", 0))
#   return s.getsockname()[0]

ip = "100.89.196.12"

common_env = dict(
    PYRO_SERIALIZER="pickle",
    PYRO_SERIALIZERS_ACCEPTED="pickle",
    PYRO_SOCK_REUSE="True",
    PYRO_PICKLE_PROTOCOL_VERSION="2",
    LOCAL_IP=ip,
    PYRO_IP=ip,
    LOCOBOT_IP=ip,
    ROBOT_NAME="hello_robot",
    CAMERA_NAME="hello_realsense",
)

# TODO(lshamis): Grab pip requirements from "requirements.txt"
# We skip "-f https://download.pytorch.org/whl/cpu/torch_stable.html"
# Add facebookresearch/fairo in develop mode
#
# pip_deps = open("requirements.txt").readlines()
pip_deps = [
    "Pyro4",
    "hello-robot-stretch-body",
    "hello-robot-stretch-body-tools",
    "pytransform3d",
    "cloudpickle",
    "scikit-fmm",
    "scikit-image",
    "torch",
    "torchvision",
    # develop facebookresearch/fairo
    "-e "
    + os.path.abspath(os.path.join(str(pathlib.Path(__file__).parent.resolve()), "../../../..")),
    # These are missing from the requirements file.
    "blosc",
]

shared_env = mrp.Conda.SharedEnv(
    name="hello_env",
    dependencies=["python=3.8", {"pip": pip_deps}],
)

procs = {
    "pyro": ["python", "-m", "Pyro4.naming", "-n", ip],
    "robot": ["python", "./remote_hello_robot.py", "--ip", ip],
    "realsense": ["python", "./remote_hello_realsense.py", "--ip", ip],
    "saver": ["python", "./remote_hello_saver.py", "--ip", ip],
    "slam": ["python", "./slam_service.py", common_env["CAMERA_NAME"]],
    "planning": ["python", "./planning_service.py"],
    "navigation": ["python", "./navigation_service.py", common_env["ROBOT_NAME"]],
}

for name, run_command in procs.items():
    mrp.process(
        name=name,
        runtime=mrp.Conda(
            shared_env=shared_env,
            run_command=run_command,
        ),
        env=common_env,
    )

mrp.main()
