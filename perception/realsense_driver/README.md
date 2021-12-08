Installation:
```
conda install -c eyeware librealsense
pip install pyrealsense2

git clone git@github.com:facebookresearch/fairo.git
cd fairo/perception/realsense_driver
pip install .
```


Usage:
```py
from realsense_wrapper import RealsenseAPI

rs = RealsenseAPI()

num_cameras = rs.get_num_cameras()
intrinsics = rs.get_intrinsices()
imgs = rs.get_images()
```