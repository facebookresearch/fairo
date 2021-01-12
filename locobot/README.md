

This folder contains a droidlet agent embodied on a [Locobot](http://www.locobot.org/). We support a Locobot Agent embodied on a physical Locobot and also in the simulation platform [Habitat](https://aihabitat.org/).

<center>

<table>
  <tr>
    <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/loco_physical.gif" width=170 height=270></td>
    <td><img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/habitat_mover.gif" width=400 height=270></td>
  </tr>
   <tr>
     <td>Locobot in the real world.</td>
     <td>Locobot being teleoperated in Habitat.</td>
   </tr>
 </table>
 
 </center>


## Setup
The Locobot Assistant is currently setup using a client-server architecture - with a thin layer on the locobot and a devserver which deals with all the heavy computation. 

**On the Locobot** 

* Setup pyrobot on the locobot using the [python 3 setup](https://github.com/facebookresearch/pyrobot/blob/master/README.md). Copy [remote_locobot.py](./remote_locobot.py) and [launch_pyro.sh](./launch_pyro.sh) to the locobot and launch the environment.

```
chmod +x launch_pyro.sh
./launch_pyro.sh  
```

**On the Devserver** 
    
```
conda create -n droidlet_env python==3.7.4 pip numpy scikit-learn==0.19.1 pytorch torchvision -c conda-forge -c pytorch
conda activate droidlet_env
cd ~/droidlet/locobot
pip install -r requirements.txt

export LOCOBOT_IP=<IP of the locobot>
```

Run with default behavior, in which agent will explore the environment
```
python locobot_agent.py
```
This will download models, datasets and spawn the dashboard that is served on `localhost:8000`.

Results should look something like this with `habitat` backend
<p align="center">
    <img src="https://media.giphy.com/media/XwmXCvoGHBXBqYUdMe/giphy.gif", width="960" height="192">
</p>

To `turn off default behaviour`
```
python locobot_agent.py --no_default_behavior
```

## ROS cheatsheet 

A set of commands to be run on the locobot to sanity-check basic functionalities. 

* rosrun tf view_frames - creates a PDF with the graph of the current transform tree to help identify different frames. These can then be used to the transformation matrices between any two frames.
* rostopic echo <topic name> (http://wiki.ros.org/rostopic) - ros publishes a stream for each activity as topics (for example one for the raw camera stream, depth stream etc). This is a useful debugging command to sanity check that the basic functionalities are working on the locobot and can help identify issues like lose cables. 

