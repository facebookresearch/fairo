# fairomsg

Cap'n Proto message definitions, auto-generated from rosmsg.

## Install

```bash
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=msg
```

## Usage

```python
import fairomsg

pkg_names = fairomsg.get_pkgs()
assert 'sensor_msgs' in pkg_names

sensor_msgs = fairomsg.get_msgs('sensor_msgs')
img_builder = sensor_msgs.Image()
```

## Build

Add ROS msg packages to the Conda environment as necessary in [msetup.py](msetup.py) by appending to `ros_msg_packages`. Then,

```bash
pip install mrp
mrp up
```
