"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import time
import queue

from droidlet.perception.robot.handlers import (
    InputHandler,
    ObjectDetection,
    FaceRecognition,
    HumanPose,
    ObjectTracking,
    MemoryHandler,
    DetectLaserPointer,
    ObjectDedup,
)
from droidlet.interpreter.robot.objects import AttributeDict
from droidlet.event import sio

from torch import multiprocessing as mp
multiprocessing = mp.get_context("spawn")


class SlowPerception:
    """Home for all slow perceptual modules used by the LocobotAgent.

    It is run as a separate process and has all the compute intensive perceptual models.

    Args:
        model_data_dir (string): path for all perception models (default: ~/locobot/agent/models/perception)
    """

    def __init__(self, model_data_dir):
        self.model_data_dir = model_data_dir
        self.vision = self.setup_vision_handlers()

    def setup_vision_handlers(self):
        """Setup all vision handlers, by defining an attribute dict of different perception handlers."""
        handlers = AttributeDict(
            {
                "detector": ObjectDetection(self.model_data_dir),
                "human_pose": HumanPose(self.model_data_dir),
                "face_recognizer": FaceRecognition(),
                "laser_pointer": DetectLaserPointer(),
                "tracker": ObjectTracking(),
            }
        )
        return handlers

    def perceive(self, rgb_depth):
        """run all percetion handlers on the current RGBDepth frame.

        Args:
            rgb_depth (RGBDepth): input frame to run all perception handlers on.

        Returns:
            rgb_depth (RGBDepth): input frame that all perception handlers were run on.
            detections (list[Detections]): list of all detections
            humans (list[Human]): list of all humans detected
        """
        detections = self.vision.detector(rgb_depth)
        humans = self.vision.human_pose(rgb_depth)
        face_detections = self.vision.face_recognizer(rgb_depth)
        if face_detections:
            detections += face_detections
        # laser_detection_obj = self.vision.laser_pointer(rgb_depth)
        # if laser_detection_obj:
        #     detections += [laser_detection_obj]
        # self.vision.tracker(rgb_depth, detections)
        return rgb_depth, detections, humans


def slow_vision_process(model_data_dir, input_queue, output_queue):
    """Set up slow vision process, consisting of compute intensive perceptual models.

    Args:
        model_data_dir (string): path for all perception models (default: ~/locobot/agent/models/perception)
        input_queue (multiprocessing.Queue): queue to retrieve input for slow vision process from
        output_queue (multiprocessing.Queue): queue to dump slow vision process output in
    """
    perception = SlowPerception(model_data_dir)

    while True:
        img, xyz = input_queue.get(block=True)
        rgb_depth, detections, humans = perception.perceive(img)
        output_queue.put([rgb_depth, detections, humans, xyz])


import traceback


class Process(multiprocessing.Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            # raise e  # You can still raise this exception if you need to

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception


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
        self.vision = self.setup_vision_handlers()
        self.send_queue = multiprocessing.Queue()
        self.recv_queue = multiprocessing.Queue()
        self.vprocess = Process(
            target=slow_vision_process,
            args=(self.model_data_dir, self.send_queue, self.recv_queue),
        )
        self.vprocess.daemon = True
        self.vprocess.start()
        self.audio = None
        self.tactile = None
        self.slow_vision_ready = True

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
                "deduplicate": ObjectDedup(),
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
            self.send_queue.put([rgb_depth, xyz])
            self.slow_vision_ready = False

        try:
            old_image, detections, humans, old_xyz = self.recv_queue.get(block=force)
            self.slow_vision_ready = True
        except queue.Empty:
            old_image, detections, humans, old_xyz = None, None, None, None

        if self.vprocess.exception:
            error, _traceback = self.vprocess.exception
            raise ChildProcessError(_traceback)

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
