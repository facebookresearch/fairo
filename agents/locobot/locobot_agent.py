"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import sys
import subprocess
import re
import numpy as np
import time
import signal
import random
import logging
import faulthandler
from multiprocessing import set_start_method

from droidlet import dashboard
if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()

from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.droidlet_nsp_model_wrapper import DroidletNSPModelWrapper
from droidlet.base_util import to_player_struct, Pos, Look, Player, hash_user
from droidlet.memory.memory_nodes import PlayerNode
from agents.loco_mc_agent import LocoMCAgent
from agents.argument_parser import ArgumentParser

from droidlet.memory.robot.loco_memory import LocoAgentMemory
from droidlet.perception import Perception, SelfPerception
from droidlet.interpreter.robot import default_behaviors

from droidlet.dialog.robot.dialogue_objects import (
    LocoBotCapabilities,
    LocoGetMemoryHandler,
    PutMemoryHandler,
    LocoInterpreter,
)
import droidlet.lowlevel.locobot.rotation as rotation
from droidlet.lowlevel.locobot.locobot_mover import LoCoBotMover
from droidlet.event import sio

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../../")
# SCHEMAS = [os.path.join(os.path.join(BASE_AGENT_ROOT, "base_agent"), "base_memory_schema.sql")]

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

    def init_event_handlers(self):
        super().init_event_handlers()

        @sio.on("command")
        def test_command(sid, commands):
            movement = [0.0, 0.0, 0.0]
            for command in commands:
                if command == "MOVE_FORWARD":
                    movement[0] += 0.1
                    print("action: FORWARD")
                elif command == "MOVE_BACKWARD":
                    movement[0] -= 0.1
                    print("action: BACKWARD")
                elif command == "MOVE_LEFT":
                    movement[2] += 0.3
                    print("action: LEFT")
                elif command == "MOVE_RIGHT":
                    movement[2] -= 0.3
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

    def init_memory(self):
        """Instantiates memory for the agent.

        Uses the DB_FILE environment variable to write the memory to a
        file or saves it in-memory otherwise.
        """
        self.memory = LocoAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            db_log_path=None,
        )
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
        self.dialogue_manager = DialogueManager(
            agent=self,
            dialogue_object_classes=dialogue_object_classes,
            opts=self.opts,
            semantic_parsing_model_wrapper=DroidletNSPModelWrapper,
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
