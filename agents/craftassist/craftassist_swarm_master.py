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

from droidlet.perception.craftassist import heuristic_perception

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

from droidlet.lowlevel.minecraft import craftassist_specs
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.craftassist.craftassist_swarm_worker import CraftAssistSwarmWorker, CraftAssistSwarmWorker_Wrapper, TASK_MAP
from droidlet.perception.craftassist.search import astar
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import COLOR_BID_MAP

import time
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
        low_level_interpreter_data = {
            'block_data': craftassist_specs.get_block_data(),
            'special_shape_functions': SPECIAL_SHAPE_FNS,
            'color_bid_map': COLOR_BID_MAP,
            'astar_search': astar,
            'get_all_holes_fn': heuristic_perception.get_all_nearby_holes}
        self.dialogue_manager = SwarmDialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
            low_level_interpreter_data=low_level_interpreter_data
        )
    
    def task_step(self, sleep_time=0.25):
        # TODO: add tag check to the query
        query = "SELECT MEMORY FROM Task WHERE prio=-1"
        _, task_mems = self.memory.basic_search(query)
        for mem in task_mems:
            if "swarm_worker_0" not in mem.get_tags():
                continue
            if mem.task.init_condition.check():
                mem.get_update_status({"prio": 0})

        # this is "select TaskNodes whose priority is >= 0 and are not paused"
        query = "SELECT MEMORY FROM Task WHERE ((prio>=0) AND (paused <= 0))"
        _, task_mems = self.memory.basic_search(query)
        for mem in task_mems:
            if "swarm_worker_0" not in mem.get_tags():
                continue
            if mem.task.run_condition.check():
                # eventually we need to use the multiplex filter to decide what runs
                mem.get_update_status({"prio": 1, "running": 1})
            if mem.task.stop_condition.check():
                mem.get_update_status({"prio": 0, "running": 0})
        # this is "select TaskNodes that are runnning (running >= 1) and are not paused"
        query = "SELECT MEMORY FROM Task WHERE ((running>=1) AND (paused <= 0))"
        _, task_mems = self.memory.basic_search(query)
        if not task_mems:
            time.sleep(sleep_time)
            return
        for mem in task_mems:
            if "swarm_worker_0" not in mem.get_tags():
                continue
            mem.task.step()
            if mem.task.finished:
                mem.update_task()

    def assign_task_to_worker(self, i, task_name, task_data):
        cur_task = TASK_MAP[task_name](self, task_data)
        self.memory.tag(cur_task.memid, "swarm_worker_{}".format(i))
        if i == 0:
            pass
        else:
            self.swarm_workers[i-1].input_tasks.put((task_name, task_data, cur_task.memid)) 

    def start(self):
        # count forever unless the shutdown signal is given
        for swarm_worker in self.swarm_workers:
            swarm_worker.start()

        while not self._shutdown:
            try:
                self.step()
                
                for i in range(self.num_agents-1):
                    flag = True
                    # TODO: implement perception memory --> handle the perceiptions queue
                    
                    while flag:
                        try:
                            name, obj = self.swarm_workers[i].query_from_worker.get_nowait()
                            if name == "task_updates":
                                for (memid, cur_task_status) in obj:
                                    mem = self.memory.get_mem_by_id(memid)
                                    mem.get_update_status({"prio": cur_task_status[0], "running": cur_task_status[1]})
                                    if cur_task_status[2]:
                                        mem.task.finished = True
                        except:
                            flag = False


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
