# Install Azure-Kinect-ROS-Driver

Please follow the [instruction here](https://github.com/facebookresearch/pyrobot/blob/Develop/robots/azure_kinect/README.md) to install Azure Kinect ROS Driver.

# Install opencv-bridge

- Create a new workspace folder and clone the opencv-bridge repo

```
mkdir $azure_bridge_ws
cd $azure_bridge_ws
catkin_init_workspace
git clone -b python3_patch_melodic https://github.com/kalyanvasudev/vision_opencv.git
```

- Activate python3 virtual env: ```source activate $virtualenv```

- Build package in virtual env

```
catkin_make --cmake-args -DPYTHON_EXECUTABLE=$(which python) -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

- Source workspace: ```source $azure_bridge_ws/devel/setup.bash```

# Run example

- Launch Azure ROS Driver

```
source $azure_kinect_ws/devel/setup.bash
roslaunch azure_kinect_ros_driver driver.launch fps:=30 color_resolution:=720P depth_mode:=NFOV_UNBINNED
```

- Run tracker

```
python ycb_azure_ros_single_object.py
```
