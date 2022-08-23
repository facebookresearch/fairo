# Polygrasp

## Installation

```bash
git submodule update --init --recursive .
mrp up --norun
```

## Download weights

```bash
./scripts/download_weights.sh
```

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

### ROS support

On the stretch run:
```
roslaunch home_robot startup_stretch_hector_slam.launch
```

Then in `$SOURCE/fairo/perception/sandbox/polygrasp`:
```
mrp up
# or
# mrp down && mrp up
```

Then you can run the polygrasp script! If you want to verify data is streaming, run rviz:
```
source ~/catkin_ws/devel/setup.bash
roslaunch home_robot visualization.launch
```
You should be able to see the robot model and its location in the world.

Run the code:
```
source ~/catkin_ws/devel/setup.bash
conda activate mrp_polygrasp
python scripts/run_stretch_grasp.py
```
