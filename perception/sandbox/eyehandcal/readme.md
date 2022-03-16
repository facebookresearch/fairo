# Understanding Eye Hand Calibraton 

## Setup

```bash
conda env create -f environment.yml
conda activate eyehandcal  # or direnv allow
```


## Assumption
* Using a realsense camera
  * Camera is pre-calibrated / intrinsic parameter can be read from system
  * Camera image is un-distorted
* You have a ARTag available in hand
  * We only use ARTag for identification (actual size does not matter)
* Polymetis is available to control the robot.
* Camera coordinate is z_point forward, x right, y down

## Transformations
* Origin is aligned with Robot Base
* Paramterizetion [rx, ry, rz, x, y, z, px, py, pz]
  * camera-to-base rotvec
  * camera-to-base translation
  * marker-to-ee translation
* Projection
  ```
    [fx, 0., ppx],
    [0., fy, ppy],
    [0., 0.,  1.]])
  ```
  
## Data Collection
  1. obtain intrinscis/image from [realsense driver](../../realsense_driver/)
  2. collect (observed_marker_in_image, ee_to_base_transform) pairs
      * observed_marker_in_image  -- cv2.detectMarkers()
      * ee_to_base_transform -- forward kinematics
  
  See [cal.py](cal.py) for detail


## Issues 
* Support Franka manual mode (no need to program motion)
* Use differentiable transform instead of finite difference
  * Was hoping torchcontrol.tranform.backward() exist - Theseus?
  * TorchControl Rotation is float32 (not good enough for findiff)
    `loss(gt_param) > 1e-3`?
* ARTag based Intrinsic calibration
* Detect poor convergence case 
  * optimality condition
  * reproj error
* Intuitive Visualization	
  * Foxglove
  * Matplotlib
  * W&B?



