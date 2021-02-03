"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import sys
import subprocess
import re
import numpy as np

# python/ dir, for agent.so
BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(BASE_AGENT_ROOT)

import dldashboard

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dldashboard.start()

from base_agent.nsp_dialogue_manager import NSPDialogueManager
from loco_memory import LocoAgentMemory
from base_agent.base_util import to_player_struct, Pos, Look, Player, hash_user
from base_agent.memory_nodes import PlayerNode
from base_agent.loco_mc_agent import LocoMCAgent
from perception import Perception, SelfPerception
from base_agent.argument_parser import ArgumentParser
import default_behaviors
from locobot.agent.dialogue_objects import LocoBotCapabilities, LocoGetMemoryHandler, PutMemoryHandler, LocoInterpreter
import rotation
from locobot_mover import LoCoBotMover
from multiprocessing import set_start_method
import time
import signal
import random
import logging
import faulthandler
from dlevent import sio


SCHEMAS = [os.path.join(os.path.join(BASE_AGENT_ROOT, "base_agent"), "base_memory_schema.sql")]

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
        self.dashboard_chat = None
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

    def init_event_handlers(self):
        super().init_event_handlers()

        @sio.on("command")
        def test_command(sid, commands):
            movement = [0.0, 0.0, 0.0]
            for command in commands:
                if command == "MOVE_FORWARD":
                    movement[0] += 0.05
                    print("action: FORWARD")
                elif command == "MOVE_BACKWARD":
                    movement[0] -= 0.05
                    print("action: BACKWARD")
                elif command == "MOVE_LEFT":
                    movement[2] += 0.1
                    print("action: LEFT")
                elif command == "MOVE_RIGHT":
                    movement[2] -= 0.1
                    print("action: RIGHT")
                elif command == "PAN_LEFT":
                    self.mover.bot.set_pan(
                        self.mover.bot.get_pan() + 0.08
                    )
                elif command == "PAN_RIGHT":
                    self.mover.bot.set_pan(
                        self.mover.bot.get_pan() - 0.08
                    )
                elif command == "TILT_UP":
                    self.mover.bot.set_tilt(
                        self.mover.bot.get_tilt() - 0.08
                    )
                elif command == "TILT_DOWN":
                    self.mover.bot.set_tilt(
                        self.mover.bot.get_tilt() + 0.08
                    )
            self.mover.move_relative([movement])

        @sio.on("sendCommandToAgent")
        def send_text_command_to_agent(sid, command):
            logging.info("in send_text_command_to_agent, got the command: %r" % (command))
            agent_chat = (
                "<dashboard> " + command
            )  # the chat is coming from a player called "dashboard"
            self.dashboard_chat = agent_chat
            dialogue_manager = self.dialogue_manager
            # send back the dictionary
            logical_form = {}
            status = ""
            try:
                logical_form = dialogue_manager.get_logical_form(
                    s=command, model=dialogue_manager.model
                )
                logging.info("logical form is : %r" % (logical_form))
                status = "Sent successfully"
            except:
                logging.info("error in sending chat")
                status = "Error in sending chat"
            # update server memory
            self.dashboard_memory["chatResponse"][command] = logical_form
            self.dashboard_memory["chats"].pop(0)
            self.dashboard_memory["chats"].append({"msg": command, "failed": False})
            payload = {
                "status": status,
                "chat": command,
                "chatResponse": self.dashboard_memory["chatResponse"][command],
                "allChats": self.dashboard_memory["chats"],
            }
            sio.emit("setChatResponse", payload)

    def init_memory(self):
        """Instantiates memory for the agent.

        Uses the DB_FILE environment variable to write the memory to a
        file or saves it in-memory otherwise.
        """
        self.memory = LocoAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path="agent_memory.{}.log".format(self.name),
        )
        file_log_handler = logging.FileHandler("agent.{}.log".format(self.name))
        file_log_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_log_handler)
        logging.info("Initialized agent memory")

    def init_perception(self):
        """Instantiates all perceptual modules.

        Each perceptual module should have a perceive method that is
        called by the base agent event loop.
        """
        if not hasattr(self, "perception_modules"):
            self.perception_modules = {}
        self.perception_modules["self"] = SelfPerception(self)
        self.perception_modules["vision"] = Perception(self, self.opts.perception_model_dir)

    def init_controller(self):
        """Instantiates controllers - the components that convert a text chat to task(s)."""
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = LocoBotCapabilities
        dialogue_object_classes["interpreter"] = LocoInterpreter
        dialogue_object_classes["get_memory"] = LocoGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.dialogue_manager = NSPDialogueManager(self, dialogue_object_classes, self.opts)

    def init_physical_interfaces(self):
        """Instantiates the interface to physically move the robot."""
        self.mover = LoCoBotMover(ip=self.opts.ip, backend=self.opts.backend, use_dslam=self.opts.use_dslam)

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
        sio.emit("showAssistantReply", {'agent_reply' : "Agent: {}".format(chat)})
        self.memory.add_chat(self.memory.self_memid, chat)
        # actually send the chat, FIXME FOR HACKATHON
        # return self._cpp_send_chat(chat)

    def step(self):
        super().step()
        time.sleep(0)

    def task_step(self, sleep_time=0.0):
        super().task_step(sleep_time=sleep_time)


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Locobot", base_path)
    opts = parser.parse()

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if opts.verbose else logging.INFO)
    sh.setFormatter(log_formatter)
    logging.getLogger().addHandler(sh)
    logging.info("Info logging")
    logging.debug("Debug logging")

    # Check that models and datasets are up to date
    if not opts.dev:
        rc = subprocess.call([opts.verify_hash_script_path, "locobot"])

    set_start_method("spawn", force=True)

    sa = LocobotAgent(opts)
    sa.start()
