# fairomsg

Cap'n Proto message definitions, auto-generated from rosmsg.

## Install

```bash
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=msg
```

## Build

Add ROS msg packages to the Conda environment as necessary in [msetup.py](msetup.py) by appending to `ros_msg_packages`. Then,

```bash
pip install mrp
mrp up
```
