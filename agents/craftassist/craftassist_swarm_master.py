"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import logging
import faulthandler
from pickle import NONE
import signal
import random
import sentry_sdk
from multiprocessing import set_start_method
from collections import namedtuple
import subprocess

# `from craftassist.agent` instead of `from .` because this file is
# also used as a standalone script and invoked via `python craftassist_agent.py`

from droidlet.lowlevel.minecraft.shapes import SPECIAL_SHAPE_FNS
import droidlet.dashboard as dashboard

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    print("starting dashboard...")
    dashboard.start()

from droidlet.dialog.swarm_dialogue_manager import SwarmDialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from agents.argument_parser import ArgumentParser
from droidlet.dialog.craftassist.dialogue_objects import MCBotCapabilities
from droidlet.interpreter.craftassist import MCGetMemoryHandler, PutMemoryHandler, SwarmMCInterpreter
# from droidlet.lowlevel.minecraft.mc_agent import Agent as MCAgent

from droidlet.lowlevel.minecraft import craftassist_specs
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.craftassist.craftassist_swarm_worker import CraftAssistSwarmWorker, CraftAssistSwarmWorker_Wrapper, TASK_MAP

import pdb

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().handlers.clear()

sentry_sdk.init()  # enabled if SENTRY_DSN set in env
DEFAULT_BEHAVIOUR_TIMEOUT = 20
DEFAULT_FRAME = "SPEAKER"
Player = namedtuple("Player", "entityId, name, pos, look, mainHand")
Item = namedtuple("Item", "id, meta")

class CraftAssistSwarmMaster(CraftAssistAgent):
    default_num_agents = 3

    def __init__(self, opts):
        try:
            self.num_agents = opts.num_agents
        except:
            logging.info("Default swarm with {} agents.".format(self.default_num_agents))
            self.num_agents = self.default_num_agents
        self.swarm_workers = [CraftAssistSwarmWorker_Wrapper(opts, idx=i) for i in range(self.num_agents - 1)]

        super(CraftAssistSwarmMaster, self).__init__(opts)
    
    def init_controller(self):
        """Initialize all controllers"""
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = MCBotCapabilities
        dialogue_object_classes["interpreter"] = SwarmMCInterpreter
        dialogue_object_classes["get_memory"] = MCGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.opts.block_data = craftassist_specs.get_block_data()
        self.opts.special_shape_functions = SPECIAL_SHAPE_FNS
        # use swarm dialogue manager that does not track its own swarm behaviorss
        self.dialogue_manager = SwarmDialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
        )
    
    def assign_task_to_worker(self, i, task_name, task_data):
        if i == 0:
            TASK_MAP[task_name](self, task_data)
        else:
            self.swarm_workers[i-1].input_tasks.put((task_name, task_data))

    def start(self):
        # count forever unless the shutdown signal is given
        for swarm_worker in self.swarm_workers:
            swarm_worker.start()

        while not self._shutdown:
            try:
                self.step()
            except Exception as e:
                self.handle_exception(e)

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()

    logging.basicConfig(level=opts.log_level.upper())

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    # Check that models and datasets are up to date and download latest resources.
    # Also fetches additional resources for internal users.
    if not opts.dev:
        rc = subprocess.call([opts.verify_hash_script_path, "craftassist"])

    set_start_method("spawn", force=True)

    sa = CraftAssistSwarmMaster(opts)
    sa.start()
