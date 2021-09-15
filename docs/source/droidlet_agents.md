```eval_rst
.. _droidlet_agents:
```
# Agents
We instantiate a droidlet [agent](https://github.com/facebookresearch/fairo/tree/agent_docs/locobot/agent) on a [Locobot](http://www.locobot.org/) and an [agent](https://github.com/facebookresearch/fairo/tree/agent_docs/craftassist/agent) in Minecraft using the [Craftassist](https://arxiv.org/abs/1907.08584) framework (the droidlet project evolved from Craftassist).  

## Locobot ##

### Locobot Perception ###

We have a high-level pipeline that runs many of the perception handlers that exist.
This pipeline is split into `Perception` and `SlowPerception`.

#### Pipelines

`Perception` only consists of the fast processing modules, that can be run in the main thread and likely need to be run at a high frequency.
`SlowPerception` consists of heavier modules such as Face Recognition, Object Detection and can be run at a much lower frequency in a background thread.

```eval_rst
 .. autoclass:: locobot.agent.perception.Perception
    :members: perceive, setup_vision_handlers, log
 .. autoclass:: locobot.agent.perception.SlowPerception
    :members: perceive, setup_vision_handlers
```

#### Components

These pipelines are powered by components that can be stringed together in arbitrary ways, to create your own custom pipeline:

```eval_rst
 .. autoclass:: locobot.agent.perception.handlers.InputHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.DetectionHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.HumanPoseHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.FaceRecognitionHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.LaserPointerHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.TrackingHandler
    :members:
 .. autoclass:: locobot.agent.perception.handlers.MemoryHandler
    :members:
```

#### Data Structures

The components use some data structure classes to create metadata such as object information and have convenient functions registered on these classes

```eval_rst
 .. autoclass:: locobot.agent.perception.RGBDepth
    :members:
 .. autoclass:: locobot.agent.perception.WorldObject
    :members:
 .. autoclass:: locobot.agent.perception.Human
    :members:
 .. autoclass:: locobot.agent.perception.HumanKeypoints
    :members:
 .. autoclass:: locobot.agent.perception.Detection
    :members:
```


### Locobot PyRobot interface ###

We have a `RemoteLocobot` object that runs on the robot, and marshals data back and forth from the robot to the server.
Additionally, on the server-side, we have a `LoCoBotMover` class that communicates with `RemoteLocobot` and provides a low-level API to the robot.

```eval_rst
 .. autoclass:: locobot.robot.RemoteLocobot
    :members:
 .. autoclass:: locobot.agent.locobot_mover.LoCoBotMover
    :members:
```

## Craftassist ##
Details for setting up and running the Cuberite server and Craftassist agent are [here](https://github.com/facebookresearch/fairo/tree/agent_docs/craftassist/)

### Craftassist Perception ###

The craftassist perception modules are mostly heuristic.  

```eval_rst
 .. autoclass:: craftassist.agent.low_level_perception.LowLevelMCPerception
   :members: perceive
 .. autoclass:: craftassist.agent.heuristic_perception.PerceptionWrapper
   :members: perceive
```

However, there are voxel models for semantic segmentation, one is [here](https://github.com/facebookresearch/fairo/tree/agent_docs/craftassist/agent/voxel_models/detection-transformer).  Its interface is:
```eval_rst
  .. autoclass:: craftassist.agent.voxel_models.subcomponent_classifier.SubcomponentClassifierWrapper
    :members: perceive
```
