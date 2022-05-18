# Polygrasp

## Installation

In a new conda environment, [install Polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html#simple).

Install the package itself:

```bash
pip install -e ../../../msg ../../realsense_driver/
pip install -e .
```

Then, follow the instructions in the forked repos to create the necessary isolated conda environments:

1. [UnseenObjectClustering](https://github.com/1heart/UnseenObjectClustering)
1. [graspnet-baseline](https://github.com/1heart/graspnet-baseline)

## Usage

Make necessary configuration modifications. For example:
- [conf/run_grasp.yaml](./conf/run_grasp.yaml) contains the configuration, e.g. robot IP.
- [conf/masks/](./conf/masks/) contains a folder to define workspace masks for each camera, for each bin.
- We expect calibrated camera parameters out of [eyehandcal](../eyehandcal).


Ensure Polymetis is running on the machine connected to the robot. Then, start the necessary pointcloud/grasping/gripper servers:

```bash
mrp up
```

The example script to bring everything together and execute the grasps:

```bash
python scripts/run_grasp.py  # Connect to robot, gripper, servers; run grasp
```

This continuously grasps from bin 1 to bin 2 until there are no more objects detected in the bin 1 workspace; then it moves the objects back from bin 2 to bin 1, and repeats.

### Mocked data

To test without a robot or cameras, run

```bash
python scripts/run_grasp.py robot=robot_mock cam=cam_mock
```

which runs the loop without connecting to a real robot and loading the RGBD images from [data/rgbd.npy](data/rgbd.npy).
