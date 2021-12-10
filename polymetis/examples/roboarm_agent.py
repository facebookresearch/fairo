"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import time
import signal
import random
import logging
import faulthandler
from multiprocessing import set_start_method
import shutil

import numpy as np
from PIL import Image
import base64
import cv2

import matplotlib.pyplot as plt
import imageio
from PIL import Image

import Pyro4

from droidlet import dashboard
#from droidlet.tools.data_scripts.try_download import try_download_artifacts

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
from droidlet.base_util import to_player_struct, Pos, Look, Player
from agents.droidlet_agent import DroidletAgent
from agents.argument_parser import ArgumentParser
from droidlet.memory.robot.loco_memory import LocoAgentMemory, DetectedObjectNode
from droidlet.perception.semantic_parsing.utils.interaction_logger import InteractionLogger

from droidlet.interpreter.robot import (
    dance,
    default_behaviors
)
from droidlet.dialog.robot import LocoBotCapabilities
import droidlet.lowlevel.rotation as rotation
from droidlet.event import sio
from droidlet.shared_data_structs import RGBDepth
from droidlet.lowlevel.robot_mover_utils import xyz_canonical_coords_to_pyrobot_coords

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()

Pyro4.config.SERIALIZER = "serpent"
Pyro4.config.SERIALIZERS_ACCEPTED.add("serpent")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


class RoboarmAgent(DroidletAgent):
    """Implements an instantiation of the LocoMCAgent . It starts
    off the agent processes including launching the dashboard.

    Args:
        opts (argparse.Namespace): opts returned by the ArgumentParser with defaults set
            that you can override.
        name (string, optional): a name for your agent (default: Roboarm)

    Example:
        >>> python roboarm_agent.py --backend 'polysim'
    """

    coordinate_transforms = rotation

    def __init__(self, opts, name="RoboArm"):
        self.backend = opts.backend
        super(RoboarmAgent, self).__init__(opts)
        logging.info("RoboarmAgent.__init__ started")
        self.agent_type = "roboarm"
        self.opts = opts
        self.entityId = 0
        self.no_default_behavior = opts.no_default_behavior
        self.last_chat_time = -1000000000000
        self.name = name
        self.player = Player(100, name, Pos(0, 0, 0), Look(0, 0))
        self.pos = Pos(0, 0, 0)
        self.uncaught_error_count = 0
        self.last_task_memid = None
        self.point_targets = []
        self.init_event_handlers()

        uri = "PYRO:obj_3695c59817ec4be3bcb56f93fbb18d5d@localhost:40243"            
        self.simEnv = Pyro4.Proxy(uri)
        self.sim_image = []
        

        
        # list of (prob, default function) pairs
        if self.backend == 'habitat':
            self.visible_defaults = [(1.0, default_behaviors.explore)]
        elif self.backend == 'polysim':
            self.visible_defaults = [(1.0, default_behaviors.explore)] 
        else:
            raise RuntimeError("Unknown backend specified {}" % (self.backend, ))
        self.interaction_logger = InteractionLogger()
        if os.path.exists("annotation_data/rgb"):
            shutil.rmtree("annotation_data/rgb")
        if os.path.exists("annotation_data/seg"):
            shutil.rmtree("annotation_data/seg")
            

    def init_event_handlers(self):
        super().init_event_handlers()


        @sio.on("movement command")
        def test_command(sid, commands, movement_values):
            if len(movement_values) == 0:
                movement_values["yaw"] = 0.01
                movement_values["velocity"] = 0.1

            #Get velocity    
            vel = movement_values["velocity"]


            for command in commands:
                if command == "PAN_LEFT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() + 0.08)
                elif command == "PAN_RIGHT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() - 0.08)
                elif command == "TILT_UP":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() - 0.08)
                elif command == "TILT_DOWN":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() + 0.08)
                elif command == "MOVE_JOINT_1":
                    #print("JOINT_1 BUTTON PRESSED")
                    self.mover.move(1,vel)
                elif command == "MOVE_JOINT_2":
                    #print("JOINT_2 BUTTON PRESSED")
                    self.mover.move(2,vel)
                elif command == "MOVE_JOINT_3":
                    #print("JOINT_3 BUTTON PRESSED")
                    self.mover.move(3,vel)    
                elif command == "MOVE_JOINT_4":
                    #print("JOINT_4 BUTTON PRESSED")
                    self.mover.move(4,vel)
                elif command == "MOVE_JOINT_5":
                    #print("JOINT_5 BUTTON PRESSED")
                    self.mover.move(5,vel)
                elif command == "MOVE_JOINT_6":
                    #print("JOINT_6 BUTTON PRESSED")
                    self.mover.move(6,vel) 
                elif command == "GO_HOME":
                    self.mover.go_home()



                    #print (self.env.get_state())
                #elif command == "GO_HOME":
                    #self.mover.go_home()                       
                elif command == "GET_POS":
                    pos = self.mover.get_ee_pos()
                    movement_values["ee_pos"] = pos
                elif command == "GET_IMAGE":
                    rgb = self.update_sim_image()
                    sio.emit("updateImage", rgb)


                    ##arrayList = self.simEnv.get_image()
                    ##npArray = np.asarray(arrayList)
                    ##npArray = npArray.astype(np.uint8)
                    #print(type(npArray))
                    #pil_image = Image.fromarray(npArray)
                    #pil_image.show()
                    #im = np.array(imageio.imread('Lenna_(test_image).png'))
                    ##im = npArray[:,:,0:3]
                    
                    ##print(im.shape)
                    ##print(type(im))

                    ##print(npArray.shape)
                    ##print(type(npArray))                    
                    ##rgb_im = self.createRGB(im)
                    #print(type(rgb_im))  
                    ##serialized_image = rgb_im.to_struct(224,10)
                    ##rgb = serialized_image["rgb"]
                    ##sio.emit("updateImage", rgb)
                    #sio.emit ("rgb", rgb) 
              
            #print(movement_values)
            #sio.emit("updatePosState", movement_values)

        @sio.on("shutdown")
        def _shutdown(sid, data):
            self.shutdown()

        @sio.on("get_memory_objects")
        def objects_in_memory(sid):
            objects = DetectedObjectNode.get_all(self.memory)
            for o in objects:
                del o["feature_repr"]  # pickling optimization
            self.dashboard_memory["objects"] = objects
            sio.emit("updateState", {"memory": self.dashboard_memory})

        @sio.on("interaction data")
        def log_interaction_data(sid, interactionData):
            self.interaction_logger.logInteraction(interactionData)

        # Returns an array of objects with updated masks
        @sio.on("label_propagation")
        def label_propagation(sid, postData):
            objects = LP.label_propagation(postData)
            sio.emit("labelPropagationReturn", objects)

#@sio.on.emit("movement command", commands, movementValues);
    def update_sim_image(self):
        arrayList = self.simEnv.get_image()
        npArray = np.asarray(arrayList)
        npArray = npArray.astype(np.uint8)
        im = npArray[:,:,0:3]                  
        rgb_im = self.createRGB(im)
        serialized_image = rgb_im.to_struct(224,10)
        rgb = serialized_image["rgb"]
        return rgb

    def init_memory(self):
        """Instantiates memory for the agent.

        Uses the DB_FILE environment variable to write the memory to a
        file or saves it in-memory otherwise.
        """
        self.memory = LocoAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path=None,
            coordinate_transforms=self.coordinate_transforms,
        )
        dance.add_default_dances(self.memory)
        logging.info("Initialized agent memory")


    def init_perception(self):
        """Instantiates all perceptual modules.

        Each perceptual module should have a perceive method that is
        called by the base agent event loop.
        """
        if not hasattr(self, "perception_modules"):
            self.perception_modules = {}
        #self.perception_modules["language_understanding"] = NSPQuerier(self.opts, self)
        #self.perception_modules["self"] = SelfPerception(self)
        #self.perception_modules["vision"] = Perception(self.opts.perception_model_dir)
	

	
    def perceive(self, force=False):
        # 1. perceive from NLU parser
        print("PERCIEVE_DEBUG")
        super().perceive(force=force)
        # 2. perceive from robot perception modules
        #self.perception_modules["self"].perceive(force=force)
        #rgb_depth = self.mover.get_rgb_depth()
        #xyz = self.mover.get_base_pos_in_canonical_coords()
        #x, y, yaw = xyz
        #if self.backend == 'habitat':
        #    sio.emit(
        #        "map",
        #        {"x": x, "y": y, "yaw": yaw, "map": self.mover.get_obstacles_in_canonical_coords()},
        #    )

        #previous_objects = DetectedObjectNode.get_all(self.memory)
        # perception_output is a namedtuple of : new_detections, updated_detections, humans
        #perception_output = self.perception_modules["vision"].perceive(rgb_depth,
        #                                                       xyz,
        #                                                       previous_objects,
        #                                                       force=force)
        #self.memory.update(perception_output)


    def init_controller(self):
        """Instantiates controllers - the components that convert a text chat to task(s)."""
        dialogue_object_classes = {}
        print("CONTROLLER_DEBUG")
        dialogue_object_classes["bot_capabilities"] = {"task": LocoBotCapabilities, "data": {}}
        #dialogue_object_classes["interpreter"] = LocoInterpreter
        #dialogue_object_classes["get_memory"] = LocoGetMemoryHandler
        #dialogue_object_classes["put_memory"] = PutMemoryHandler
        #self.dialogue_manager = DialogueManager(
        #    memory=self.memory,
        #    dialogue_object_classes=dialogue_object_classes,
        #    dialogue_object_mapper=DialogueObjectMapper,
        #    opts=self.opts,
        #)

    def init_physical_interfaces(self):
        """Instantiates the interface to physically move the robot."""
        print(f"IP::: PHYSICAL INTERFACE {self.opts.ip}")
        self.mover = Pyro4.core.Proxy('PYRO:remotefranka@' + self.opts.ip + ':9090')

    def step(self):
        #super().step()
        time.sleep(0)
        rgb = self.update_sim_image()
        sio.emit("updateImage", rgb)
        

    def task_step(self, sleep_time=0.0):
        super().task_step(sleep_time=sleep_time)

    def shutdown(self):
        self._shutdown = True
        try:
            self.perception_modules["vision"].vprocess_shutdown.set()
        except:
            """
            the try/except is there in the event that
            self.perception_modules["vision"] has either:
            1. not been fully started yet
            2. already crashed / shutdown due to other effects
            """
            pass
        time.sleep(5)  # let the other threads die
        os._exit(0)  # TODO: remove and figure out why multiprocess sometimes hangs on exit

    def createRGB (self,im):
        p = im.shape[0]
        xs = np.sort(np.random.uniform(0, 10, p))
        xs = np.tile(xs, (p, 1))

        ys = np.sort(np.random.uniform(0, 10, p))
        ys[::-1].sort()
        ys = np.transpose(np.tile(ys, (p,1)))

        # (x,y,z=1) in row-major form, in locobot coords
        pts = np.asarray([xyz_canonical_coords_to_pyrobot_coords((x,y,1))
            for x,y in zip(xs.ravel(), ys.ravel())])
        
        depth = np.ones((p, p))
        #rgb = np.float32(np.random.rand(p, p, 3) * 255)
        rgb = im

        rgb_d = RGBDepth(rgb, depth, pts)
        return rgb_d


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Roboarm", base_path)
    
    opts = parser.parse()
    print (opts.ip, opts.backend)
    opts.ip = "172.23.42.96"
    print(f"IP::: Roboarm AGENT {opts.ip}")

    logging.basicConfig(level=opts.log_level.upper())
    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    # Check that models and datasets are up to date
    #if not opts.dev:
     #   try_download_artifacts(agent="locobot")

    #set_start_method("spawn", force=True)

    sa = RoboarmAgent(opts)
    sa.start()

