"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import faulthandler
import signal
import random
import sentry_sdk
from multiprocessing import set_start_method
from collections import namedtuple

from droidlet.perception.craftassist import heuristic_perception

from droidlet.lowlevel.minecraft.shapes import SPECIAL_SHAPE_FNS
import droidlet.dashboard as dashboard
from droidlet.tools.artifact_scripts.try_download import try_download_artifacts

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    print("starting dashboard...")
    dashboard.start()

from droidlet.dialog.swarm_dialogue_manager import SwarmDialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from agents.argument_parser import ArgumentParser
from droidlet.dialog.craftassist.mc_dialogue_task import MCBotCapabilities
from droidlet.interpreter.craftassist import (
    MCGetMemoryHandler,
    PutMemoryHandler,
    SwarmMCInterpreter,
)

from droidlet.lowlevel.minecraft import craftassist_specs
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.craftassist.craftassist_swarm_worker import CraftAssistSwarmWorker_Wrapper, TASK_MAP
from droidlet.perception.craftassist.search import astar
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import COLOR_BID_MAP
from droidlet.memory.memory_nodes import (  # noqa
    TaskNode,
    TripleNode,
    PlayerNode,
    ProgramNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    ReferenceObjectNode,
    NamedAbstractionNode,
    AttentionNode,
    NODELIST,
)
from droidlet.interpreter.craftassist.tasks import *

import time
import pickle

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


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except:
        return False
    return True


class empty_object:
    def __init__(self) -> None:
        pass


class CraftAssistSwarmMaster(CraftAssistAgent):
    default_num_agents = 2

    def __init__(self, opts):
        try:
            self.num_agents = opts.num_agents
        except:
            logging.info("Default swarm with {} agents.".format(self.default_num_agents))
            self.num_agents = self.default_num_agents
        self.swarm_workers = [
            CraftAssistSwarmWorker_Wrapper(opts, idx=i) for i in range(1, self.num_agents)
        ]

        super(CraftAssistSwarmMaster, self).__init__(opts)

    def init_controller(self):
        """Initialize all controllers"""
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = {"task": MCBotCapabilities, "data": {}}
        dialogue_object_classes["interpreter"] = SwarmMCInterpreter
        dialogue_object_classes["get_memory"] = MCGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        self.opts.block_data = craftassist_specs.get_block_data()
        self.opts.special_shape_functions = SPECIAL_SHAPE_FNS
        low_level_interpreter_data = {
            "block_data": craftassist_specs.get_block_data(),
            "special_shape_functions": SPECIAL_SHAPE_FNS,
            "color_bid_map": COLOR_BID_MAP,
            "astar_search": astar,
            "get_all_holes_fn": heuristic_perception.get_all_nearby_holes,
        }
        self.dialogue_manager = SwarmDialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
            low_level_interpreter_data=low_level_interpreter_data,
        )

        self.handle_query_dict = {
            "_db_read": self.memory._db_read,
            "_db_read_one": self.memory._db_read_one,
            "_db_write": self.memory._db_write,
            "db_write": self.memory.db_write,
            "tag": self.memory.tag,
            "untag": self.memory.untag,
            "forget": self.memory.forget,
            "add_triple": self.memory.add_triple,
            "get_triples": self.memory.get_triples,
            "check_memid_exists": self.memory.check_memid_exists,
            "get_mem_by_id": self.memory.get_mem_by_id,
            "basic_search": self.memory.basic_search,
            "get_block_object_by_xyz": self.memory.get_block_object_by_xyz,
            "get_block_object_ids_by_xyz": self.memory.get_block_object_ids_by_xyz,
            "get_object_info_by_xyz": self.memory.get_object_info_by_xyz,
            "get_block_object_by_id": self.memory.get_block_object_by_id,
            "get_object_by_id": self.memory.get_object_by_id,
            "get_instseg_object_ids_by_xyz": self.memory.get_instseg_object_ids_by_xyz,
            "upsert_block": self.memory.upsert_block,
            "_update_voxel_count": self.memory._update_voxel_count,
            "_update_voxel_mean": self.memory._update_voxel_mean,
            "remove_voxel": self.memory.remove_voxel,
            "set_memory_updated_time": self.memory.set_memory_updated_time,
            "set_memory_attended_time": self.memory.set_memory_attended_time,
            "add_chat": self.memory.add_chat,
        }

    def if_swarm_task(self, mem):
        for i in range(1, self.num_agents):
            if "swarm_worker_{}".format(i) in mem.get_tags():
                return True
        return False

    def task_step(self, sleep_time=0.25):
        # TODO: add tag check to the query
        query = "SELECT MEMORY FROM Task WHERE prio=-1"
        _, task_mems = self.memory.basic_search(query)
        for mem in task_mems:
            if not self.if_swarm_task(mem):
                if mem.task.init_condition.check():
                    mem.get_update_status({"prio": 0})

        # this is "select TaskNodes whose priority is >= 0 and are not paused"
        query = "SELECT MEMORY FROM Task WHERE ((prio>=0) AND (paused <= 0))"
        _, task_mems = self.memory.basic_search(query)
        for mem in task_mems:
            if not self.if_swarm_task(mem):
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
            if not self.if_swarm_task(mem):
                mem.task.step()
                if mem.task.finished:
                    mem.update_task()

    def assign_task_to_worker(self, i, task_name, task_data):
        if i == 0:
            TASK_MAP[task_name](self, task_data)
        else:
            self.swarm_workers[i - 1].input_tasks.put((task_name, task_data, None))

    def handle_memory_query(self, query):
        query_id = query[0]
        query_name = query[1]
        query_args = query[2:]
        if query_name in self.handle_query_dict.keys():
            to_return = self.handle_query_dict[query_name](*query_args)
        else:
            logging.info("swarm master cannot handle memory query: {}".format(query))
            raise NotImplementedError
        to_return = self.safe_object(to_return)
        return tuple([query_id, to_return])

    def safe_single_object(self, input_object):
        if is_picklable(input_object):
            return input_object
        all_attrs = dir(input_object)
        return_obj = empty_object()
        for attr in all_attrs:
            if attr.startswith("__"):
                continue
            if type(getattr(input_object, attr)).__name__ in dir(__builtins__):
                setattr(return_obj, attr, getattr(input_object, attr))
        return return_obj

    def get_safe_single_object_attr_dict(self, input_object):
        return_dict = {}
        all_attrs = vars(input_object)
        for attr in all_attrs:
            if attr.startswith("__"):
                continue
            if is_picklable(getattr(input_object, attr)):
                return_dict[attr] = all_attrs[attr]
        return return_dict

    def safe_object(self, input_object):
        if isinstance(input_object, tuple):
            tuple_len = len(input_object)
            to_return = []
            for i in range(tuple_len):
                to_return.append(self.safe_single_object(input_object[i]))
            return tuple(to_return)
        else:
            return self.safe_single_object(input_object)

    def get_new_tasks(self, tag):
        query = "SELECT MEMORY FROM Task WHERE prio=-1"
        _, task_mems = self.memory.basic_search(query)
        task_list = []
        for mem in task_mems:
            if tag not in mem.get_tags():
                continue
            else:
                task_name = mem.task.__class__.__name__.lower()
                task_data = self.get_safe_single_object_attr_dict(mem.task)
                memid = mem.task.memid
                task_list.append((task_name, task_data, memid))
        return task_list

    def step_assign_new_tasks_to_workers(self):
        for i in range(self.num_agents - 1):
            task_list = self.get_new_tasks(tag="swarm_worker_{}".format(i + 1))
            for new_task in task_list:
                self.swarm_workers[i].input_tasks.put(new_task)

    def step_update_tasks_with_worker_data(self):
        # task updates xinfo from swarm worker process
        for i in range(self.num_agents - 1):
            flag = True
            while flag:
                if self.swarm_workers[i].query_from_worker.empty():
                    flag = False
                else:
                    name, obj = self.swarm_workers[i].query_from_worker.get_nowait()
                    if name == "task_updates":
                        for (memid, cur_task_status) in obj:
                            mem = self.memory.get_mem_by_id(memid)
                            mem.get_update_status(
                                {"prio": cur_task_status[0], "running": cur_task_status[1]}
                            )
                            if cur_task_status[2]:
                                mem.task.finished = True
                    elif name == "initialization":
                        self.init_status[i] = True

    def step_handle_worker_memory_queries(self):
        for i in range(self.num_agents - 1):
            flag = True
            while flag:
                if self.swarm_workers[i].memory_send_queue.empty():
                    flag = False
                else:
                    query = self.swarm_workers[i].memory_send_queue.get_nowait()
                    response = self.handle_memory_query(query)
                    self.swarm_workers[i].memory_receive_queue.put(response)

    def start(self):
        # count forever unless the shutdown signal is given
        for swarm_worker in self.swarm_workers:
            swarm_worker.start()

        self.init_status = [False] * (self.num_agents - 1)
        while not self._shutdown:
            try:
                if all(self.init_status):
                    self.step()
                self.step_assign_new_tasks_to_workers()
                self.step_update_tasks_with_worker_data()
                self.step_handle_worker_memory_queries()

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
        try_download_artifacts(agent="craftassist")

    set_start_method("spawn", force=True)

    sa = CraftAssistSwarmMaster(opts)
    sa.start()
