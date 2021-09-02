import os
import logging
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.loco_mc_agent import LocoMCAgent
from agents.craftassist.swarm_utils import get_safe_single_object_attr_dict, safe_object
from droidlet.interpreter.craftassist import tasks
from droidlet.dialog.craftassist.dialogue_objects import MCBotCapabilities
from droidlet.interpreter.craftassist import MCGetMemoryHandler, PutMemoryHandler, SwarmMCInterpreter
from droidlet.lowlevel.minecraft import craftassist_specs
from droidlet.lowlevel.minecraft.shapes import SPECIAL_SHAPE_FNS
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import COLOR_BID_MAP
from droidlet.perception.craftassist.search import astar
from droidlet.perception.craftassist import heuristic_perception
from droidlet.dialog.swarm_dialogue_manager import SwarmDialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from multiprocessing import Process, Queue, set_start_method
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception
from droidlet.memory.craftassist.swarm_worker_memory import SwarmWorkerMemory
from droidlet.lowlevel.minecraft.mc_util import MCTime
from agents.argument_parser import ArgumentParser
import subprocess
import pdb, sys

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

TASK_MAP =  {
        "move": tasks.Move,
        "build": tasks.Build,
        "destroy": tasks.Destroy,
        "dig": tasks.Dig,       
    }
TASK_INFO = {
    "move": ["target"],
    "build": ["blocks_list"],
    "destroy": ["schematic"],
    "dig": ["origin", "length", "width", "depth"]
}


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class SwarmMasterWrapper():
    def __init__(self, base_agent, worker_agents, opts) -> None:
        self.base_agent = base_agent
        self.agent_type = base_agent.__class__.__name__.lower()
        self.opts = opts
        self.init_workers(worker_agents, opts)
        self.init_master_controller()
        self.init_memory_handlers_dict()
        base_agent.task_step_filters = ["swarm_worker_{}".format(i+1) for i in range(self.num_workers)]
        base_agent.num_agents = self.num_workers + 1

    def init_workers(self, worker_agents, opts):
        self.num_workers = len(worker_agents)
        self.swarm_workers = [SwarmWorkerWrapper(opts, idx=i+1) for i in range(self.num_workers)]

    def init_master_controller(self):
        if 'craft' in self.agent_type:
            dialogue_object_classes = {}
            dialogue_object_classes["bot_capabilities"] = MCBotCapabilities
            dialogue_object_classes["interpreter"] = SwarmMCInterpreter
            dialogue_object_classes["get_memory"] = MCGetMemoryHandler
            dialogue_object_classes["put_memory"] = PutMemoryHandler
            self.base_agent.opts.block_data = craftassist_specs.get_block_data()
            self.base_agent.opts.special_shape_functions = SPECIAL_SHAPE_FNS
            low_level_interpreter_data = {
                'block_data': craftassist_specs.get_block_data(),
                'special_shape_functions': SPECIAL_SHAPE_FNS,
                'color_bid_map': COLOR_BID_MAP,
                'astar_search': astar,
                'get_all_holes_fn': heuristic_perception.get_all_nearby_holes}
            self.base_agent.dialogue_manager = SwarmDialogueManager(
                memory=self.base_agent.memory,
                dialogue_object_classes=dialogue_object_classes,
                dialogue_object_mapper=DialogueObjectMapper,
                opts=self.base_agent.opts,
                low_level_interpreter_data=low_level_interpreter_data
            )
        elif 'loco' in self.agent_type:
            # TODO: implement for locobot
            pass
        else:
            logging.info("agent type not implemented for swarm")
            raise NotImplementedError
        
    def init_memory_handlers_dict(self):
        if 'craft' in self.agent_type:
            self.handle_query_dict = {
                "_db_read": self.base_agent.memory._db_read,
                "_db_read_one": self.base_agent.memory._db_read_one,
                "_db_write": self.base_agent.memory._db_write,
                "db_write": self.base_agent.memory.db_write,
                "tag": self.base_agent.memory.tag,
                "untag": self.base_agent.memory.untag,
                "forget": self.base_agent.memory.forget,
                "add_triple": self.base_agent.memory.add_triple,
                "get_triples": self.base_agent.memory.get_triples,
                "check_memid_exists": self.base_agent.memory.check_memid_exists,
                "get_mem_by_id": self.base_agent.memory.get_mem_by_id,
                "basic_search": self.base_agent.memory.basic_search,
                "get_block_object_by_xyz": self.base_agent.memory.get_block_object_by_xyz,
                "get_block_object_ids_by_xyz": self.base_agent.memory.get_block_object_ids_by_xyz,
                "get_object_info_by_xyz": self.base_agent.memory.get_object_info_by_xyz,
                "get_block_object_by_id": self.base_agent.memory.get_block_object_by_id,
                "get_object_by_id": self.base_agent.memory.get_object_by_id,
                "get_instseg_object_ids_by_xyz": self.base_agent.memory.get_instseg_object_ids_by_xyz,
                "upsert_block": self.base_agent.memory.upsert_block,
                "_update_voxel_count": self.base_agent.memory._update_voxel_count,
                "_update_voxel_mean": self.base_agent.memory._update_voxel_mean,
                "remove_voxel": self.base_agent.memory.remove_voxel,
                "set_memory_updated_time": self.base_agent.memory.set_memory_updated_time,
                "set_memory_attended_time": self.base_agent.memory.set_memory_attended_time,
                "add_chat": self.base_agent.memory.add_chat
            }
        elif 'loco' in self.agent_type:
            # TODO: implement for locobot
            pass
        else:
            logging.info("agent type not implemented for swarm")
            raise NotImplementedError

    def get_new_tasks(self, tag):
        query = "SELECT MEMORY FROM Task WHERE prio=-1"
        _, task_mems = self.base_agent.memory.basic_search(query)
        task_list = []
        for mem in task_mems:
            if tag not in mem.get_tags():
                continue
            else:
                task_name = mem.task.__class__.__name__.lower()
                task_data = get_safe_single_object_attr_dict(mem.task)
                memid = mem.task.memid
                task_list.append((task_name, task_data, memid))
        return task_list

    def assign_new_tasks_to_workers(self):
        for i in range(self.num_workers):
            task_list = self.get_new_tasks(tag="swarm_worker_{}".format(i+1))
            for new_task in task_list:
                self.swarm_workers[i].input_tasks.put(new_task)

    def update_tasks_with_worker_data(self):
        for i in range(self.num_workers):
            flag = True
            while flag:
                if self.swarm_workers[i].query_from_worker.empty():
                    flag = False
                else:
                    name, obj = self.swarm_workers[i].query_from_worker.get_nowait()
                    if name == "task_updates":
                        for (memid, cur_task_status) in obj:
                            mem = self.base_agent.memory.get_mem_by_id(memid)
                            mem.get_update_status({"prio": cur_task_status[0], "running": cur_task_status[1]})
                            if cur_task_status[2]:
                                mem.task.finished = True
                    elif name == "initialization":
                        self.init_status[i] = True

    def handle_worker_memory_queries(self):
        for i in range(self.num_workers):
            flag = True
            while flag:
                if self.swarm_workers[i].memory_send_queue.empty():
                    flag = False
                else:
                    query = self.swarm_workers[i].memory_send_queue.get_nowait()
                    response = self.handle_memory_query(query)
                    self.swarm_workers[i].memory_receive_queue.put(response)

    def handle_memory_query(self, query):
        query_id = query[0]
        query_name = query[1]
        query_args = query[2:]
        if query_name in self.handle_query_dict.keys():
            to_return = self.handle_query_dict[query_name](*query_args)
        else:
            logging.info("swarm master cannot handle memory query: {}".format(query))
            raise NotImplementedError
        to_return = safe_object(to_return)
        return tuple([query_id, to_return])

    def start(self):
        # count forever unless the shutdown signal is given
        for swarm_worker in self.swarm_workers:
            swarm_worker.start()

        self.init_status = [False] * (self.num_workers)
        while not self.base_agent._shutdown:
            try:
                if all(self.init_status):
                    # TODO: enable adding filters for general task step
                    self.base_agent.step()
                self.assign_new_tasks_to_workers()
                self.update_tasks_with_worker_data()
                self.handle_worker_memory_queries()
                                
            except Exception as e:
                self.base_agent.handle_exception(e)


class SwarmWorkerWrapper(Process):
    def __init__(self, opts, idx=0) -> None:
        super().__init__()
        self.opts = opts
        self.idx = idx
        self.input_tasks = Queue()
        self.perceptions = Queue()
        self.query_from_worker = Queue()
        self.query_from_master = Queue()
        self.memory_send_queue = Queue()
        self.memory_receive_queue = Queue()
        self.stop = 0
    
    def init_worker(self, agent):
        self.agent_type = agent.__class__.__name__.lower()

        agent.agent_idx = self.idx
        agent.task_stacks = dict()
        agent.task_ghosts = []
        agent.prio = dict()
        agent.running = dict()
        agent.pause = dict()
        agent.memory_send_queue = self.memory_send_queue
        agent.memory_receive_queue = self.memory_receive_queue
        agent.query_from_worker = self.query_from_worker

        # perception
        if 'craft' in self.agent_type:
            agent.perception_modules = {}
            agent.perception_modules["low_level"] = SwarmLowLevelMCPerception(agent)

        elif 'loco' in self.agent_type:
            # TODO: implement for locobot
            pass
        else:
            logging.info("agent type not implemented for swarm")
            raise NotImplementedError

        # memory
        if 'craft' in self.agent_type:
            agent.memory = SwarmWorkerMemory(agent_time=MCTime(agent.get_world_time),
                                             memory_send_queue=self.memory_send_queue,
                                             memory_receive_queue=self.memory_receive_queue,
                                             memory_tag="swarm_worker_{}".format(agent.agent_idx))
        elif 'loco' in self.agent_type:
            # TODO: implement for locobot
            pass
        else:
            logging.info("agent type not implemented for swarm")
            raise NotImplementedError
        
        # controller
        if 'craft' in self.agent_type:
            agent.disable_chat = True
    
    def check_task_info(self, task_name, task_data):
        if task_name not in TASK_INFO.keys():
            logging.info("task {} received without checking arguments")
            return True
        for key in TASK_INFO[task_name.lower()]:
            if key not in task_data:
                return False
        return True

    def preprocess_data(self, task_name, task_data):
        if "task_data" in task_data:
            return task_data["task_data"]
        else:
            return task_data

    def send_queries(self, queries):
        # TODO: send queries to master by pushing to self.query_from_worker
        pass

    def send_task_updates(self, task_updates):
        """send task updates to master by pushing to self.query_from_worker
        """
        if len(task_updates)>0:
            name = 'task_updates'
            self.query_from_worker.put((name, task_updates))

    def handle_input_task(self, agent):
        flag = True
        while flag:
            if self.input_tasks.empty():
                flag = False
            else:
                task_class_name, task_data, task_memid = self.input_tasks.get_nowait()
                if task_memid is None or ((task_memid not in agent.task_stacks.keys()) and (task_memid not in agent.task_ghosts)):
                    task_data = self.preprocess_data(task_class_name, task_data)
                    if self.check_task_info(task_class_name, task_data):
                        new_task = TASK_MAP[task_class_name](agent, task_data)
                        if task_memid is None:
                            task_memid = new_task.memid
                        else:
                            agent.task_ghosts.append(new_task.memid)
                            # can send updates back to main agent to mark as finished
                        agent.task_stacks[task_memid] = new_task
                        agent.prio[task_memid] = -1
                        agent.running[task_memid] = -1
                        agent.pause[task_memid] = False
                elif task_memid in agent.task_stacks.keys():
                    self.send_task_updates([(task_memid, (agent.prio[task_memid], agent.running[task_memid], agent.task_stacks[task_memid].finished))])
                elif task_memid in agent.task_ghosts:
                    self.send_task_updates([(task_memid, (0, 0, True))])
    
    def handle_master_query(self, agent):
        flag = True
        while flag:
            if self.query_from_master.empty():
                flag = False
            else:
                query_name, query_data = self.query_from_master.get_nowait()
                if query_name == "stop":
                    for memid, task in agent.task_stacks.items():
                        if not task.finished:
                            agent.pause[memid] = True
                elif query_name == "resume":
                    for memid, task in agent.task_stacks.items():
                        agent.pause[memid] = False
                else:
                    logging.info("Query not handled: {}".format(query_name))
                    raise NotImplementedError

    def perceive(self, agent, force=False):
        for v in agent.perception_modules.values():
            v.perceive(force=force)

    def task_step(self, agent):
        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if agent.prio[memid] == -1:
                if task.init_condition.check():
                    agent.prio[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if (not agent.pause[memid]) and (agent.prio[memid] >=0):
                if task.run_condition.check():
                    agent.prio[memid] = 1
                    agent.running[memid] = 1
                if task.stop_condition.check():
                    agent.prio[memid] = 0
                    agent.running[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)    
            if (not agent.pause[memid]) and (agent.running[memid] >=1):
                tmp_query = task.step()
                # TODO: return queries if anything is missing
                if task.finished:
                    finished_task_memids.append(memid)
                    cur_task_status = (0, 0, task.finished)
                if tmp_query is not None:
                    queries.append((memid, tmp_query))
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        for memid in finished_task_memids:
            del agent.task_stacks[memid]
            del agent.prio[memid]
            del agent.running[memid]
        return queries, task_updates

    def run(self):
        agent = CraftAssistAgent(self.opts)
        self.init_worker(agent)
        self.query_from_worker.put(("initialization", True))
        while True:
            worker_perception = self.perceive(agent)
            self.perceptions.put(worker_perception)
            
            self.handle_input_task(agent)
            queries, _ = self.task_step(agent)
            self.handle_master_query(agent)
            self.send_queries(queries)
            agent.count += 1

def test_mc_swarm():
    num_workers = 1
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
    sa = CraftAssistAgent(opts)
    master = SwarmMasterWrapper(sa, [None] * num_workers, opts)
    master.start()

if __name__ == "__main__":
    test_mc_swarm()
