# Polygrasp

## Installation

In a new conda environment, [install Polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html#simple).

Install the package itself:

```bash
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


Start the necessary pointcloud/grasping/gripper servers:

```bash
mrp up
```

The script to bring everything together and execute the grasps:

```bash
python scripts/run_grasp.py  # Connect to robot, gripper, servers; run grasp
```

This continuously grasps from one bin to the other until there are no more objects detected in the workspace; then it moves the objects the other direction.
