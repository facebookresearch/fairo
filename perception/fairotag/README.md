# ArUcoX

[![CircleCI](https://circleci.com/gh/fair-robotics/fair-aruco/tree/master.svg?style=svg&circle-token=f1e087bc2f213a5f47dde7e0d67a91c1e5882294)](https://circleci.com/gh/fair-robotics/fair-aruco/tree/master)

**Note: This documentation is out of date! Newer version coming soon.**

A thin wrapper around [ArUco](https://www.uco.es/investiga/grupos/ava/node/26) that provides a cleaner API and default parameters that work with pre-generated calibration boards and markers.
Performs detection and pose estimation of ARTag markers.

The source code for the API can be found at [python/arucoX](https://github.com/fair-robotics/fair-aruco/blob/master/python/arucoX)

## Requirements
- numpy
- sophuspy
- opencv-contrib-python
- gtsam (requires dev version, which is available as a conda package in the channel `fair-robotics`)

## Installation
Install dependencies
```
conda install -c fair-robotics -c conda-forge cmake pybind11 gtsam
```
(Note 1: Cmake and Pybind11 are required to build wheel for sophuspy)
(Note 2: Cmake and Pybind11 are required to build wheel for sophuspy)

Install from source
```
git clone git@github.com:facebookresearch/fairo.git
cd fairo/perception/fairotag
pip install .
```

## Usage: Single camera

Example can be found in [examples/single_camera](https://github.com/fair-robotics/fair-aruco/blob/master/examples/single_camera)

### Creating Markers

The calibration board can be found in `resources/calib.io_charuco_215.9x279.4_7x5_35_DICT_4X4.pdf`.

Some sample markers are given in `resources/markers`.

Additional markers can be generated [here](https://chev.me/arucogen/).

### Camera Calibration
Calibration can be performed by printing the provided calibration board pdf on a letter paper (8.5" x 11"), then taking photos of the calibration board from different angles. 
As seen in the example, calibration can be performed even if the calibration board partially occluded in the images.

<p align="center">
  <img src="examples/single_camera/figs/charuco_1.jpg" width="120" align="middle">
  <img src="examples/single_camera/figs/charuco_2.jpg" width="120" align="middle">
  <img src="examples/single_camera/figs/charuco_3.jpg" width="120" align="middle">
  <img src="examples/single_camera/figs/charuco_4.jpg" width="120" align="middle">
  <img src="examples/single_camera/figs/charuco_5.jpg" width="120" align="middle">
</p>

```py
import fairotag as frt

c = frt.CameraModule()
c.calibrate_camera(calib_img_list)
```

`calib_img_list`: list of images of the calibration board taken from different angles.

Camera calibration parameters can also be saved/loaded by:
```py
# Saving/loading from dict
params = c.get_intrinsics()
c.set_intrinsics(params)

# Saving/loading from file
c.save_intrinsics(filename)
c.load_intrinsics(filename)
```
(`params` is a dictionary containing fields `"camera_matrix"` and `"distortion_coeffs"`.)

(NOTE: Calibration with checkerboard is buggy. Importing intrinsics from the camera driver is recommended.)

### Marker Detection
Input image `img`:
<p align="center">
  <img src="examples/single_camera/figs/test_5x5.jpg" width="360" align="middle">
</p>

```py
# Register marker lengths to enable pose estimation of corresponding markers
c.register_marker_size(0, MARKER_LENGTH)
c.register_marker_size(3, MARKER_LENGTH)
c.register_marker_size(4, MARKER_LENGTH)

# Perform detection and pose estimation
markers = c.detect_markers(img, marker_length=MARKER_LENGTH)
```

`MARKER_LENGTH`: Width of the printed markers.

`markers`: A list of `MarkerInfo` - a NamedTuple containing the following fields:
- `id`: Marker ID
- `corner`: Image coordinates of the four corners of the marker 
- `length`: Marker length (`None` if marker not registered)
- `pose`: Transformation of the marker in the camera frame expressed using [sophus](https://pypi.org/project/sophuspy/).SE3  (`None` if marker not registered)

Note: The following helper functions are provided to translate `sophus.SE3` into quaternions:
```py
import sophus as sp

t = sp.SE3()

quat = frt.utils.so3_to_quat(t.so3())      # array([0., 0., 0., 1.])
pos, quat = frt.utils.se3_to_pos_quat(t)   # array([0., 0., 0.]), array([0., 0., 0., 1.])
```

### Rendering Results

Identified markers and their estimated poses can also be rendered on the input image for visualization and debug purposes.

```py
# [Option 1] Detect and render markers on image
img_rend = c.render_markers(img)

# [Option 2] Save computation time by providing previous output from "detect_markers"
img_rend = c.render_markers(img, markers)
```
Output image `img_rend`:
<p align="center">
  <img src="examples/single_camera/figs/test_5x5_render.jpg" width="360" align="middle">
</p>

## Usage: Multi-camera scene

Documentation in progress

## Notes
- More precise measurements of the marker length will result in better pose estimations.
- Markers require white spaces around them to be detected.

## Todos
Major:
- Multi-camera scene documentation
- Hand-eye coordination

Minor:
- Improve pose estimation by not constructing the full graph every call
- Resolve potential corner case when moving camera is not connected to world frame through observations