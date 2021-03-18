# Basic SLAM


## To run on habitat

- launch roscore

```
roscore &
```

- In other terminal, launch the slam test

```
load_pyrobot_env
python slam.py --robot habitat --goal 10 2 0 --map_size 4000 --robot_rad 25 --vis --save_vis --store_path ./tmp --dataset_path /Replica-Dataset --scene apartment_0
```

 Results should look something like this
<p align="center">
    <img src="https://media.giphy.com/media/eTof1wrCZHXw83tGwH/giphy.gif", width="960" height="192">
</p>

- `save_vis` arg will also store `rgb, depth, seg, robot trajectory(data.json)` under `store_path(args)/scene(args)` folder
- To run for any other scene apart from `apartment_0` provide the appropriate `dataset-path` & `scene` args

## To run on locobot

- launch the camera and robot base (make sure base is turned `on`)

 ```
roslaunch locobot_control main.launch use_base:=true use_camera:=true
```

- In other terminal, launch the slam test

```
load_pyrobot_env
python slam.py --robot locobot --goal 4 0 0 --map_size 1000 --robot_rad 25 --save_vis
```

Results should look something like this
<p align="center">
    <img src="https://media.giphy.com/media/sjWJMYAF3NYRXyJj5o/giphy.gif",  width="960" height="">
</p>

## Helpful commands

- for converting images to video

```
ffmpeg -framerate 6 -f image2 -i %04d.jpg out.gif
```

## Collect active vision data on real robot
All the things will run on robot
- launch the camera and robot base (make sure base is turned `on`)

 ```
roslaunch locobot_control main.launch use_base:=true use_camera:=true
```

- run the agent

```
python slam_locobot.py --robot locobot --goal 9.5 0 0 --map_size 2000 --robot_rad 25 --save_vis
```

- data will be stored under `tmp` folder 