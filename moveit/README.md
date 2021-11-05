# Moveit Bridge API

Simple API that exposes planning functionalities of [MoveIt](https://moveit.ros.org/)

## Requirements

- docker
- torch
- polymetis

## Installation
```
pip install -e .
```

## Example Usage
1. Run the MoveIt planning server:
```
cd moveit_server
sudo ./run.sh
```

2. In a separate terminal, run the Polymetis server (this example uses the simulation robot client)
```
launch_robot.py robot_client=franka_sim gui=false use_real_time=false
```

2. In a third terminal, run the example script (or any user script)
```
python example.py
```