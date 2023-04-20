# Moveit Bridge API

Simple API that exposes planning functionalities of [MoveIt](https://moveit.ros.org/)

## Functionalities
- [x] Querying MoveIt for a joint trajectory plan given current joint state and desired end-effector pose
- [x] Updating MoveIt scene with obstacles from pybullet to plan for obstacle avoidance

## Requirements

- docker
- torch
- [Polymetis](https://polymetis-docs.github.io/)

### Note: Docker setup
Ensure user account is part of the `docker` group after the installation of Docker by:
```
sudo groupadd -f docker
sudo usermod -aG docker $USER
```
then logging out and back in.

## Installation
```
pip install -e .
```

## Example Usage
1. Run the MoveIt planning server:
```
cd moveit_server
./run.sh
```

2. In a separate terminal, run the Polymetis server (this example uses the simulation robot client)
```
launch_robot.py robot_client=franka_sim gui=false use_real_time=false
```

2. In a third terminal, run the example script (or any user script)
```
python example.py
```

## Documentation
Moveit planning interface methods accessible through `MoveitInterface`:
- `add_mesh(name, mesh_pos, mesh_quat, filename)`
- `attach_mesh(link, name, mesh_pos, mesh_quat, filename)`
- `remove_attached_object(link, name)`
- `remove_world_object(name)`

Documentation for the methods can be found [here](https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/planning_scene_interface.py).