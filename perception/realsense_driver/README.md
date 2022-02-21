Installation:
```
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/realsense_driver
```

Optional conda environment
```
conda create -n eyehandcal polymetis librealsense -c fair-robotics
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/realsense_driver
```

Usage:
```py
from realsense_wrapper import RealsenseAPI

rs = RealsenseAPI()

num_cameras = rs.get_num_cameras()
intrinsics = rs.get_intrinsics()
imgs = rs.get_images()
```