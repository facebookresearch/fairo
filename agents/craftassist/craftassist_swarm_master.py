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
from agents.craftassist.craftassist_swarm_worker import CraftAssistSwarmWorker

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
        self.swarm_workers = [CraftAssistSwarmWorker(opts, idx=i) for i in range(self.num_agents)]

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
            swarm_workers_names=self.names,
            opts=self.opts,
        )
    
    def task_step(self, sleep_time=0.25):
        super().task_step(sleep_time)
        for worker in self.swarm_workers:
            worker.task_step(sleep_time)

    def perceive(self, force=False):
        """Whenever some blocks are changed, that area will be put into a 
        buffer which will be force-perceived by the agent in the next step

        Here the agent first clusters areas that are overlapping on the buffer,
        then run through all perception modules to perceive
        and finally clear the buffer when perception is done.
        """
        super().perceive()
        for worker in self.swarm_workers:
            worker.perceive()

    def send_chat(self, chat, agent_idx=None, agent_name=None):
        """Send chat from agent to player"""
        agent_idx = self.get_agent_idx(agent_idx, agent_name)
        if agent_idx is None:
            return
        self.memory.add_chat(self.memory.self_memid, chat)
        return self.swarm_workers[agent_idx].send_chat(chat)

    # TODO update client so we can just loop through these
    # TODO rename things a bit- some perceptual things are here,
    #      but under current abstraction should be in init_perception
    def init_physical_interfaces(self):
        """Initializes the physical interfaces of the agent."""
        # For testing agent without cuberite server

        if self.opts.port == -1:
            return
        logging.info("Attempting to connect to port {}".format(self.opts.port))
        
        # create a swarm of agents
        self.names = []
        self.cagents = []
        for i in range(self.num_agents):
            self.names.append(self.swarm_workers[i].name)
            self.cagents.append(self.swarm_workers[i].cagent)

        # agent 0 is the master agent, the others are worker/sub agents
        self.cagent = self.cagents[0]

        self.dig = self.cagent.dig
        self.drop_item_stack_in_hand = self.cagent.drop_item_stack_in_hand
        self.drop_item_in_hand = self.cagent.drop_item_in_hand
        self.drop_inventory_item_stack = self.cagent.drop_inventory_item_stack
        self.set_inventory_slot = self.cagent.set_inventory_slot
        self.get_player_inventory = self.cagent.get_player_inventory
        self.get_inventory_item_count = self.cagent.get_inventory_item_count
        self.get_inventory_items_counts = self.cagent.get_inventory_items_counts
        # defined above...
        # self.send_chat = self.cagent.send_chat
        self.set_held_item = self.cagent.set_held_item
        self.step_pos_x = self.cagent.step_pos_x
        self.step_neg_x = self.cagent.step_neg_x
        self.step_pos_z = self.cagent.step_pos_z
        self.step_neg_z = self.cagent.step_neg_z
        self.step_pos_y = self.cagent.step_pos_y
        self.step_neg_y = self.cagent.step_neg_y
        self.step_forward = self.cagent.step_forward
        self.look_at = self.cagent.look_at
        self.set_look = self.cagent.set_look
        self.turn_angle = self.cagent.turn_angle
        self.turn_left = self.cagent.turn_left
        self.turn_right = self.cagent.turn_right
        self.place_block = self.cagent.place_block
        self.use_entity = self.cagent.use_entity
        self.use_item = self.cagent.use_item
        self.use_item_on_block = self.cagent.use_item_on_block
        self.is_item_stack_on_ground = self.cagent.is_item_stack_on_ground
        self.craft = self.cagent.craft
        self.get_blocks = self.cagent.get_blocks
        self.get_local_blocks = self.cagent.get_local_blocks
        self.get_incoming_chats = self.get_chats
        self.get_player = self.cagent.get_player
        self.get_mobs = self.cagent.get_mobs
        self.get_other_players = self.get_all_players
        self.get_other_player_by_name = self.cagent.get_other_player_by_name
        self.get_vision = self.cagent.get_vision
        self.get_line_of_sight = self.cagent.get_line_of_sight
        self.get_player_line_of_sight = self.get_all_player_line_of_sight
        self.get_changed_blocks = self.cagent.get_changed_blocks
        self.get_item_stacks = self.cagent.get_item_stacks
        self.get_world_age = self.cagent.get_world_age
        self.get_time_of_day = self.cagent.get_time_of_day
        self.get_item_stack = self.cagent.get_item_stack

    def get_agent_idx(self, agent_idx, agent_name):
        """
        general handler for deciding which is the agent we are interested in
        if agent_idx is not None, return it
        otherwise, return the agent_idx based on agent_name 
        """
        if agent_idx is None:
            if agent_name is None:
                agent_idx = 0
            else:
                try:
                    agent_idx = self.names.index(agent_name)
                except:
                    logging.info("agent name not found.")
                    # return None
                    return 0
        return agent_idx

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
