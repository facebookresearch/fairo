# FairoTag

A wrapper around [ArUco](https://www.uco.es/investiga/grupos/ava/node/26) and [GTSAM](https://gtsam.org/) that provides minimal API to:
- Perform detection and pose estimation of ARTag markers.
- Calibrate camera extrinsics.
- Pose estimation of multiple cameras and markers within the same scene.

## Requirements
- numpy
- sophuspy
- opencv-contrib-python
- gtsam

## Installation
Install dependencies
```
conda install -c conda-forge cmake pybind11
```
(Note: Cmake and Pybind11 are required to build wheel for sophuspy)

Install from source
```
git clone git@github.com:facebookresearch/fairo.git
cd fairo/perception/fairotag
pip install .
```

## Usage

See [tutorials](tutorials/) to learn about the API and usage.

## Marker Resources

The calibration board can be found in `resources/calib.io_charuco_215.9x279.4_7x5_35_DICT_4X4.pdf`.

Some sample markers are given in `resources/markers`.

Additional markers can be generated [here](https://chev.me/arucogen/).

## Todos
- Develop a working solution of hand-and-eye coordination
- Improve pose estimation by not constructing the full graph every call
- Resolve potential corner case when moving camera is not connected to world frame through observations
