```eval_rst
.. _droidlet_agents:
```
# Agents
We instantiate a droidlet [agent](https://github.com/facebookresearch/fairo/tree/main/agents/locobot) on a [Locobot](http://www.locobot.org/) and an [agent](https://github.com/facebookresearch/fairo/tree/main/agents/craftassist) in Minecraft using the [Craftassist](https://arxiv.org/abs/1907.08584) framework (the droidlet project evolved from Craftassist).  

## Locobot ##

### Locobot Perception ###

We have a high-level pipeline that runs many of the perception handlers that exist.
This pipeline is split into `Perception` and `SlowPerception`.

#### Pipelines

`Perception` only consists of the fast processing modules, that can be run in the main thread and likely need to be run at a high frequency.
`SlowPerception` consists of heavier modules such as Face Recognition, Object Detection and can be run at a much lower frequency in a background thread.

```eval_rst
 .. autoclass:: droidlet.perception.robot.perception.Perception
    :members: perceive, setup_vision_handlers, log
 
```

#### Components

These pipelines are powered by components that can be stringed together in arbitrary ways, to create your own custom pipeline:

```eval_rst
 .. autoclass:: droidlet.perception.robot.handlers.detector.ObjectDetection
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.human_pose.HumanPose
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.face_recognition.FaceRecognition
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.laser_pointer.DetectLaserPointer
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.tracker.ObjectTracking
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.error_sampling.ErrorSampler
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.label_propagate.LabelPropagate
```

#### Data Structures

The components use some data structure classes to create metadata such as object information and have convenient functions registered on these classes

```eval_rst
 .. autoclass:: droidlet.shared_data_structs.RGBDepth
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.core.WorldObject
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.human_pose.Human
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.human_pose.HumanKeypoints
    :members:
 .. autoclass:: droidlet.perception.robot.handlers.detector.Detection
    :members:
```


### Locobot PyRobot interface ###

We have a `RemoteLocobot` object that runs on the robot, and marshals data back and forth from the robot to the server.
Additionally, on the server-side, we have a `LoCoBotMover` class that communicates with `RemoteLocobot` and provides a low-level API to the robot.

```eval_rst
 .. autoclass:: droidlet.lowlevel.locobot.remote.remote_locobot.RemoteLocobot
    :members:
 .. autoclass:: droidlet.lowlevel.locobot.locobot_mover.LoCoBotMover
    :members:
```

## Craftassist ##
Details for setting up and running the Cuberite server and Craftassist agent are [here](https://github.com/facebookresearch/fairo/tree/main/agents/craftassist)

### Craftassist Perception ###

The craftassist perception modules are mostly heuristic.  

```eval_rst
 .. autoclass:: droidlet.perception.craftassist.low_level_perception.LowLevelMCPerception
   :members: perceive
 .. autoclass:: droidlet.perception.craftassist.heuristic_perception.PerceptionWrapper
   :members: perceive
```

However, there are voxel models for semantic segmentation, one is [here](https://github.com/facebookresearch/fairo/tree/main/droidlet/perception/craftassist/voxel_models/detection-transformer).  Its interface is:
```eval_rst
  .. autoclass:: droidlet.perception.craftassist.voxel_models.subcomponent_classifier.SubcomponentClassifierWrapper
    :members: perceive
```
