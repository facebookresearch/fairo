"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import subprocess
import time
import signal
import random
import logging
import faulthandler
from multiprocessing import set_start_method
import shutil

from droidlet import dashboard
if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from droidlet.base_util import to_player_struct, Pos, Look, Player
from droidlet.memory.memory_nodes import PlayerNode
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from agents.loco_mc_agent import LocoMCAgent
from agents.argument_parser import ArgumentParser
import agents.locobot.label_prop as LP
from droidlet.memory.robot.loco_memory import LocoAgentMemory, DetectedObjectNode
from droidlet.perception.robot import Perception
from droidlet.perception.semantic_parsing.utils.interaction_logger import InteractionLogger
from self_perception import SelfPerception
from droidlet.interpreter.robot import (
    dance, 
    default_behaviors,
    LocoGetMemoryHandler, 
    PutMemoryHandler, 
    LocoInterpreter,
)
from droidlet.dialog.robot import LocoBotCapabilities
import droidlet.lowlevel.locobot.rotation as rotation
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
from droidlet.event import sio

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()


class LocobotAgent(LocoMCAgent):
    """Implements an instantiation of the LocoMCAgent on a Locobot. It starts
    off the agent processes including launching the dashboard.

    Args:
        opts (argparse.Namespace): opts returned by the ArgumentParser with defaults set
            that you can override.
        name (string, optional): a name for your agent (default: Locobot)

    Example:
        >>> python locobot_agent.py --backend 'locobot'
    """

    coordinate_transforms = rotation

    def __init__(self, opts, name="Locobot"):
        super(LocobotAgent, self).__init__(opts)
        logging.info("LocobotAgent.__init__ started")
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
        # list of (prob, default function) pairs
        self.visible_defaults = [(1.0, default_behaviors.explore)]
        self.interaction_logger = InteractionLogger()
        if os.path.exists("annotation_data/rgb"): 
            shutil.rmtree("annotation_data/rgb")
        if os.path.exists("annotation_data/seg"): 
            shutil.rmtree("annotation_data/seg")
        
    def init_event_handlers(self):
        super().init_event_handlers()

        @sio.on("movement command")
        def test_command(sid, commands, movement_values={}):
            if len(movement_values) == 0: 
                movement_values["yaw"] = 0.01
                movement_values["velocity"] = 0.1

            movement = [0.0, 0.0, 0.0]
            for command in commands:
                if command == "MOVE_FORWARD":
                    movement[0] += movement_values["velocity"]
                    print("action: FORWARD")
                elif command == "MOVE_BACKWARD":
                    movement[0] -= movement_values["velocity"]
                    print("action: BACKWARD")
                elif command == "MOVE_LEFT":
                    movement[2] += movement_values["yaw"]
                    print("action: LEFT")
                elif command == "MOVE_RIGHT":
                    movement[2] -= movement_values["yaw"]
                    print("action: RIGHT")
                elif command == "PAN_LEFT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() + 0.08)
                elif command == "PAN_RIGHT":
                    self.mover.bot.set_pan(self.mover.bot.get_pan() - 0.08)
                elif command == "TILT_UP":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() - 0.08)
                elif command == "TILT_DOWN":
                    self.mover.bot.set_tilt(self.mover.bot.get_tilt() + 0.08)
            self.mover.move_relative([movement])

        @sio.on("shutdown")
        def _shutdown(sid, data):
            self.shutdown()

        @sio.on("get_memory_objects")
        def objects_in_memory(sid):
            objects = DetectedObjectNode.get_all(self.memory)
            for o in objects:
                del o["feature_repr"] # pickling optimization
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
        
        @sio.on("save_rgb_seg")
        def save_rgb_seg(sid, postData): 
            LP.save_rgb_seg(postData)
            if "callback" in postData and postData["callback"]: 
                sio.emit("saveRgbSegCallback")

        @sio.on("save_annotations")
        def save_annotations(sid, categories): 
            LP.save_annotations(categories)


        @sio.on("save_categories_properties")
        def save_categories_properties(sid, categories, properties): 
            LP.save_categories_properties(categories, properties)

        @sio.on("retrain_detector")
        def retrain_detector(sid, settings={}): 
            inference_json = LP.retrain_detector(settings)
            sio.emit("annotationRetrain", inference_json)

        @sio.on("switch_detector")
        def switch_detector(sid): 
            model_dir = "annotation_data/model"
            model_names = os.listdir(model_dir)
            model_nums = list(map(lambda x: int(x.split("v")[1]), model_names))
            last_model_num = max(model_nums) 
            model_path = os.path.join(model_dir, "v" + str(last_model_num))
            detector_weights = "model_999.pth"
            properties_file = "props.json"
            things_file = "things.json"

            files = os.listdir(model_path)
            if detector_weights not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return
            if properties_file not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return
            if things_file not in files: 
                print("Error switching model:", os.path.join(model_path, things_file), "not found")
                return

            print("switching to", model_path)
            self.perception_modules["vision"] = Perception(model_path, default_keypoints_path=True)


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
        self.chat_parser = NSPQuerier(self.opts)
        if not hasattr(self, "perception_modules"):
            self.perception_modules = {}
        self.perception_modules["self"] = SelfPerception(self)
        self.perception_modules["vision"] = Perception(self.opts.perception_model_dir)

    def perceive(self, force=False):
        super().perceive(force=force, parser_only=True)
        self.perception_modules["self"].perceive(force=force)
        rgb_depth = self.mover.get_rgb_depth()
        xyz = self.mover.get_base_pos_in_canonical_coords()
        x, y, yaw = xyz
        sio.emit("map", {
            "x": x,
            "y": y,
            "yaw": yaw,
            "map": self.mover.get_obstacles_in_canonical_coords()
        })

        previous_objects = DetectedObjectNode.get_all(self.memory)
        new_state = self.perception_modules["vision"].perceive(rgb_depth,
                                                               xyz,
                                                               previous_objects,
                                                               force=force)
        if new_state is not None:
            new_objects, updated_objects = new_state
            for obj in new_objects:
                obj.save_to_memory(self.memory)
            for obj in updated_objects:
                obj.save_to_memory(self.memory, update=True)

    def init_controller(self):
        """Instantiates controllers - the components that convert a text chat to task(s)."""
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = LocoBotCapabilities
        dialogue_object_classes["interpreter"] = LocoInterpreter
        dialogue_object_classes["get_memory"] = LocoGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.dialogue_manager = DialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
        )

    def init_physical_interfaces(self):
        """Instantiates the interface to physically move the robot."""
        self.mover = LoCoBotMover(ip=self.opts.ip, backend=self.opts.backend)

    def get_player_struct_by_name(self, speaker_name):
        p = self.memory.get_player_by_name(speaker_name)
        if p:
            return p.get_struct()
        else:
            return None

    def get_other_players(self):
        return [self.player]

    def get_incoming_chats(self):
        all_chats = []
        speaker_name = "dashboard"
        if self.dashboard_chat is not None:
            if not self.memory.get_player_by_name(speaker_name):
                PlayerNode.create(
                    self.memory,
                    to_player_struct((None, None, None), None, None, None, speaker_name),
                )
            all_chats.append(self.dashboard_chat)
            self.dashboard_chat = None
        return all_chats

    # # FIXME!!!!
    def send_chat(self, chat: str):
        logging.info("Sending chat: {}".format(chat))
        # Send the socket event to show this reply on dashboard
        sio.emit("showAssistantReply", {"agent_reply": "Agent: {}".format(chat)})
        self.memory.add_chat(self.memory.self_memid, chat)
        # actually send the chat, FIXME FOR HACKATHON
        # return self._cpp_send_chat(chat)

    def step(self):
        super().step()
        time.sleep(0)

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
        time.sleep(5) # let the other threads die
        os._exit(0) # TODO: remove and figure out why multiprocess sometimes hangs on exit


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Locobot", base_path)
    opts = parser.parse()

    logging.basicConfig(level=opts.log_level.upper())
    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    # Check that models and datasets are up to date
    if not opts.dev:
        rc = subprocess.call([opts.verify_hash_script_path, "locobot"])

    set_start_method("spawn", force=True)

    sa = LocobotAgent(opts)
    sa.start()
