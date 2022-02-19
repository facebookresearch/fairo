Installation:
```
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/realsense_driver
```


Usage:
```py
from realsense_wrapper import RealsenseAPI

rs = RealsenseAPI()

num_cameras = rs.get_num_cameras()
intrinsics = rs.get_intrinsices()
imgs = rs.get_images()
```