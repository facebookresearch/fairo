# Basic SLAM


## To run on habitat

- launch roscore

```
roscore &
```

- In other terminal, launch the slam test

```
load_pyrobot_env
python slam.py --robot habitat --goal 10 2 0 --map_size 4000 --robot_rad 25 --vis --save_vis 
```

 Results should look something like this
<p align="center">
    <img src="https://media.giphy.com/media/eTof1wrCZHXw83tGwH/giphy.gif", width="960" height="192">
</p>
 
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
