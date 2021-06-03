"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import time

from droidlet.multiprocess import BackgroundTask
from droidlet.perception.robot.handlers import (
    InputHandler,
    ObjectDetection,
    FaceRecognition,
    HumanPose,
    ObjectTracking,
    MemoryHandler,
    DetectLaserPointer,
    ObjectDeduplicator,
)
from droidlet.interpreter.robot.objects import AttributeDict
from droidlet.event import sio
import queue


class Perception:
    """Home for all perceptual modules used by the LocobotAgent.

    It provides a multiprocessing mechanism to run the more compute intensive perceptual
    models (for example our object detector) as a separate process.

    Args:
        agent (LocoMCAgent): reference to the LocobotAgent
        model_data_dir (string): path for all perception models (default: ~/locobot/agent/models/perception)
    """

    def __init__(self, agent, model_data_dir):
        self.model_data_dir = model_data_dir
        self.agent = agent

        def slow_perceive_init(weights_dir):
            print(weights_dir)
            return AttributeDict(
                {
                    "detector": ObjectDetection(weights_dir),
                    "human_pose": HumanPose(weights_dir),
                    "face_recognizer": FaceRecognition(),
                    "tracker": ObjectTracking(),
                })

        def slow_perceive(models, rgb_depth, xyz):
            detections = models.detector(rgb_depth)
            humans = models.human_pose(rgb_depth)
            face_detections = models.face_recognizer(rgb_depth)
            if face_detections:
                detections += face_detections
            return rgb_depth, detections, humans, xyz

        self.vprocess = BackgroundTask(init_fn=slow_perceive_init,
                                   init_args=(model_data_dir,),
                                   process_fn=slow_perceive)
        self.vprocess.start()
        self.slow_vision_ready = True

        self.vision = self.setup_vision_handlers()
        self.audio = None
        self.tactile = None

        self.log_settings = {
            "image_resolution": 512,  # pixels
            "image_quality": 10,  # from 10 to 100, 100 being best
        }

        @sio.on("update_image_settings")
        def update_log_settings(sid, new_values):
            self.log_settings["image_resolution"] = new_values["image_resolution"]
            self.log_settings["image_quality"] = new_values["image_quality"]
            sio.emit("image_settings", self.log_settings)

    def setup_vision_handlers(self):
        """Setup all vision handlers, by defining an attribute dict of different perception handlers."""
        handlers = AttributeDict(
            {
                "input": InputHandler(self.agent, read_from_camera=True),
                "deduplicate": ObjectDeduplicator(),
                "memory": MemoryHandler(self.agent),
            }
        )
        return handlers

    def perceive(self, force=False):
        """Called by the core event loop for the agent to run all perceptual
        models and save their state to memory. It fetches the results of
        SlowPerception if they are ready.

        Args:
            force (boolean): set to True to force waiting on the SlowPerception models to finish, and execute
                all perceptual models to execute sequentially (doing that is a good debugging tool)
                (default: False)
        """
        rgb_depth = self.vision.input()

        if self.slow_vision_ready:
            xyz = self.agent.mover.get_base_pos_in_canonical_coords()
            self.vprocess.put(rgb_depth, xyz)
            self.slow_vision_ready = False

        try:
            old_image, detections, humans, old_xyz = self.vprocess.get(block=force)
            self.slow_vision_ready = True
        except queue.Empty:
            old_image, detections, humans, old_xyz = None, None, None, None

        if detections is not None:
            previous_objects = self.vision.memory.get_objects()
            current_objects = detections + humans
            if previous_objects is not None:
                new_objects, updated_objects = self.vision.deduplicate(
                    current_objects, previous_objects
                )
            self.vision.memory(new_objects, updated_objects)

        self.log(rgb_depth, detections, humans, old_image, old_xyz)

    def log(self, rgb_depth, detections, humans, old_rgb_depth, old_xyz):
        """Log all relevant data from the perceptual models for the dashboard.

        All the data here is sent to the dashboard using a socketio emit event. This data
        is then used by the dashboard for different debugging visualizations. Add any data
        that you would want to fetch on the dashboard here.

        Args:
            rgb_depth (RGBDepth): the current RGBDepth frame. This frame might be different from
            old_rgb_depth, which is the RGBDepth frame for which SlowPerception has been run
            detections (list[Detections]): list of all detections
            humans (list[Human]): list of all humans detected
            old_rgb_depth (RGBDepth): RGBDepth frame for which detections and humans are being sent.
            This is the frame for which SlowPerception has been run.
            old_xyz (list[floats]): the (x,y,yaw) of the robot at the time of old_rgb_depth

        """
        if hasattr(sio, "mock"):
            return

        sio.emit("image_settings", self.log_settings)
        if old_xyz is None:
            x, y, yaw = self.agent.mover.get_base_pos_in_canonical_coords()
        else:
            x, y, yaw = old_xyz
        resolution = self.log_settings["image_resolution"]
        quality = self.log_settings["image_quality"]
        payload = {}
        payload["time"] = time.time()
        payload["image"] = rgb_depth.to_struct(resolution, quality)
        payload["object_image"] = (
            old_rgb_depth.to_struct(resolution, quality) if old_rgb_depth is not None else -1
        )
        payload["objects"] = [x.to_struct() for x in detections] if detections is not None else []
        payload["humans"] = [x.to_struct() for x in humans] if humans is not None else []
        payload["x"] = x
        payload["y"] = y
        payload["yaw"] = yaw
        payload["map"] = self.agent.mover.get_obstacles_in_canonical_coords()
        sio.emit("sensor_payload", payload)
