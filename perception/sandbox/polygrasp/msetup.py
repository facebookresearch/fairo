import os
import mrp

if "CUDA_HOME" not in os.environ:
    raise RuntimeError("Please set the CUDA_HOME environment variable to compile third_party/graspnet-baseline/pointnet2 and third_party/graspnet-baseline/knn.")

if "CONDA_PREFIX" not in os.environ:
    raise RuntimeError(f"No conda environment detected. Please activate a conda environment before running.")

pip_path = f"{os.environ['CONDA_PREFIX']}/bin/pip"

polygrasp_setup_commands = [
    [pip_path, "install", "-e", "../../../msg"],
    [pip_path, "install", "-e", "../../realsense_driver"],
    [pip_path, "install", "-e", "."],
]

mrp.process(
    name="segmentation_server",
    runtime=mrp.Conda(
        yaml="./third_party/UnseenObjectClustering/environment.yml",
        setup_commands=[
            [pip_path, "install", "-e", "./third_party/UnseenObjectClustering/"]
        ]
        + polygrasp_setup_commands,
        run_command=["python", "-m", "utils.mrp_wrapper"],
    ),
)

mrp.process(
    name="grasp_server",
    runtime=mrp.Conda(
        yaml="./third_party/graspnet-baseline/environment.yml",
        setup_commands=[
            [pip_path, "install", "./third_party/graspnet-baseline/pointnet2/"],
            [pip_path, "install", "-e", "./third_party/graspnet-baseline/"],
        ]
        + polygrasp_setup_commands,
        run_command=["python", "-m", "graspnet_baseline.mrp_wrapper"],
    ),
)

polygrasp_shared_env = mrp.Conda.SharedEnv(
    "polygrasp",
    channels=["pytorch", "fair-robotics", "aihabitat", "conda-forge"],
    dependencies=["polymetis"],
    setup_commands=polygrasp_setup_commands,
)

mrp.process(
    name="cam_pub",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=["python", "-m", "polygrasp.cam_pub_sub"],
    ),
)

mrp.process(
    name="gripper_server",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=[
            "launch_gripper.py",
            "gripper=robotiq_2f",
            "gripper.comport=/dev/ttyUSB1",
        ],
    ),
)

mrp.main()
