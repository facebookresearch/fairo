# Polygrasp

## Installation

```bash
git submodule update --init --recursive .
mrp up --norun
```

Then, follow the instructions in the forked repos under [third_party](./third_party/) to download pretrained model weights:

1. [UnseenObjectClustering](./third_party/UnseenObjectClustering/README.md)
    1. `gdown --id 1O-ymMGD_qDEtYxRU19zSv17Lgg6fSinQ -O ./third_party/UnseenObjectClustering/data/`
1. [graspnet-baseline](./third_party/graspnet-baseline/README.md)
    1. `gdown --id 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O data/`

## Usage

Make necessary configuration modifications. For example:
- [conf/run_grasp.yaml](./conf/run_grasp.yaml) contains the configuration, e.g. robot IP.
- [conf/masks/](./conf/masks/) contains a folder to define workspace masks for each camera, for each bin.
- We expect calibrated camera parameters out of [eyehandcal](../eyehandcal); if this is not calibrated, please follow instructions there.


Ensure Polymetis is running on the machine connected to the robot. Then, start the necessary pointcloud/grasping/gripper servers:

```bash
mrp up

mrp ps  # Ensure processes are alive
mrp logs --old  # Check logs
```

The example script to bring everything together and execute the grasps:

```bash
conda activate mrp_polygrasp
python scripts/run_grasp.py  # Connect to robot, gripper, servers; run grasp
```

This continuously grasps from bin 1 to bin 2 until there are no more objects detected in the bin 1 workspace; then it moves the objects back from bin 2 to bin 1, and repeats.

### Mocked data

To test without a robot or cameras, run

```bash
conda activate mrp_polygrasp
python scripts/run_grasp.py robot=robot_mock cam=cam_mock num_bin_shifts=1 num_grasps_per_bin_shift=1
```

which runs the loop without connecting to a real robot and loading the RGBD images from [data/rgbd.npy](data/rgbd.npy).
