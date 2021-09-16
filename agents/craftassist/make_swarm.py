from droidlet import dialog
import os
import logging
from typing import Dict
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.craftassist.swarm_utils import get_safe_single_object_attr_dict, safe_object
from agents.craftassist.swarm_configs import get_default_task_info, get_swarm_interpreter, get_memory_handlers_dict
from droidlet.dialog.swarm_dialogue_manager import SwarmDialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
from multiprocessing import Process, Queue
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception
from droidlet.memory.swarm_worker_memory import SwarmWorkerMemory

import pdb, sys
from copy import deepcopy


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
    def __init__(self, base_agent, worker_agents, opts, swarm_config) -> None:
        self.base_agent = base_agent
        self.agent_type = base_agent.__class__.__name__.lower()
        self.opts = opts
        self.swarm_config = swarm_config
        self.init_workers(worker_agents, opts)
        self.init_master_controller()
        self.init_memory_handlers_dict()
        
        base_agent.task_step_filters = ["swarm_worker_{}".format(i+1) for i in range(self.num_workers)]
        base_agent.num_agents = self.num_workers + 1

    def init_workers(self, worker_agents, opts):
        self.num_workers = len(worker_agents)
        task_map = self.base_agent.dialogue_manager.dialogue_object_mapper.dialogue_objects["interpreter"].task_objects
        disable_perception_modules = self.swarm_config["disable_perception_modules"]
        self.swarm_workers = [SwarmWorkerWrapper(opts, task_map=task_map, disable_perception_modules=disable_perception_modules, idx=i+1) for i in range(self.num_workers)]
        self.base_agent.swarm_workers_memid = [None for i in range(self.num_workers)]
        self.swarm_workers_memid = self.base_agent.swarm_workers_memid

    def init_master_controller(self):
        dialogue_object_classes = self.base_agent.dialogue_manager.dialogue_object_mapper.dialogue_objects
        dialogue_object_classes["interpreter"] = get_swarm_interpreter(self.base_agent)
        self.base_agent.dialogue_manager = SwarmDialogueManager(
                memory=self.base_agent.memory,
                dialogue_object_classes=dialogue_object_classes,
                dialogue_object_mapper=DialogueObjectMapper,
                opts=self.base_agent.opts,
                low_level_interpreter_data=self.base_agent.dialogue_manager.dialogue_object_mapper.low_level_interpreter_data
            )
        
    def init_memory_handlers_dict(self):
        # TODO: customized to different agent
        self.handle_query_dict = get_memory_handlers_dict(self.base_agent)
        
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

    def handle_worker_perception(self):
        tmp_perceptoins = [dict() for i in range(self.num_workers)]
        worker_eids = dict()
        for i in range(self.num_workers):
            flag = True
            while flag:
                if self.swarm_workers[i].perceptions.empty():
                    flag = False
                else:
                    eid, name, obj = self.swarm_workers[i].perceptions.get_nowait()
                    tmp_perceptoins[i][name] = obj
                    worker_eids[i] = eid
        
        # resolve conflicts code
        
                                        

        # write back to memory
        for i in range(self.num_workers):
            if i not in worker_eids.keys():
                continue
            eid = worker_eids[i]
            if "pos" in tmp_perceptoins[i].keys():
                mem = self.base_agent.memory.get_player_by_eid(eid)
                memid = mem.memid
                cmd = (
                    "UPDATE ReferenceObjects SET eid=?, x=?,  y=?, z=? WHERE "
                )
                cmd = cmd + "uuid=?"
                self.base_agent.memory.db_write(cmd, eid, tmp_perceptoins[i]["pos"].x, 
                                                tmp_perceptoins[i]["pos"].y, 
                                                tmp_perceptoins[i]["pos"].z, memid)

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
                    elif name == "memid":
                        self.swarm_workers_memid[i] = obj

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
                    self.base_agent.step()
                self.handle_worker_perception()
                self.assign_new_tasks_to_workers()
                self.update_tasks_with_worker_data()
                self.handle_worker_memory_queries()
                                
            except Exception as e:
                self.base_agent.handle_exception(e)


class SwarmWorkerWrapper(Process):
    def __init__(self, opts, task_map, disable_perception_modules, idx=0) -> None:
        super().__init__()
        self.opts = opts
        self.idx = idx
        self.input_tasks = Queue()
        self.perceptions = Queue()
        self.query_from_worker = Queue()
        self.query_from_master = Queue()
        self.memory_send_queue = Queue()
        self.memory_receive_queue = Queue()
        self.init_task_map(task_map)
        self.disable_perception_modules = disable_perception_modules
        
    def init_task_map(self, task_map, task_info=None):
        self.TASK_MAP = deepcopy(task_map)
        self.TASK_INFO = get_default_task_info(task_map)
        if task_info is not None:
            for key in task_info:
                self.TASK_INFO[key] = task_info[key]

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

        # # craftassist 
        # p = agent.get_player()
        # agent.entityId = p.entityId

        # disable perception modules
        for module_key in self.disable_perception_modules:
            del agent.perception_modules[module_key]
        
        #### temporary for debug
        agent.perception_modules = dict()
        agent.perception_modules["low_level"] = SwarmLowLevelMCPerception(agent)
        #### end temporary for debug
        
        # memory
        agent.memory = SwarmWorkerMemory(memory_send_queue=self.memory_send_queue,
                                         memory_receive_queue=self.memory_receive_queue,
                                         memory_tag="swarm_worker_{}".format(agent.agent_idx))        
        # controller
        agent.disable_chat = True
    
    def check_task_info(self, task_name, task_data):
        if task_name not in self.TASK_INFO.keys():
            logging.info("task {} received without checking arguments")
            return True
        for key in self.TASK_INFO[task_name.lower()]:
            if key not in task_data:
                return False
        return True

    def preprocess_data(self, task_name, task_data):
        if "task_data" in task_data:
            return task_data["task_data"]
        else:
            return task_data

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
                if task_class_name not in self.TASK_MAP.keys():
                    logging.info("task not understood by worker")
                    continue
                if task_memid is None or ((task_memid not in agent.task_stacks.keys()) and (task_memid not in agent.task_ghosts)):
                    task_data = self.preprocess_data(task_class_name, task_data)
                    if self.check_task_info(task_class_name, task_data):
                        new_task = self.TASK_MAP[task_class_name](agent, task_data)
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

    # TOFIX --> 
    def send_perception_updates(self, agent):
        pass


    def perceive(self, agent, force=False):
        for v in agent.perception_modules.values():
            v.perceive(force=force)

    def task_step(self, agent):
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

        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)    
            if (not agent.pause[memid]) and (agent.running[memid] >=1):
                task.step()
                if task.finished:
                    finished_task_memids.append(memid)
                    cur_task_status = (0, 0, task.finished)
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        for memid in finished_task_memids:
            del agent.task_stacks[memid]
            del agent.prio[memid]
            del agent.running[memid]
        return task_updates

    def run(self):
        agent = CraftAssistAgent(self.opts)
        self.init_worker(agent)
        self.query_from_worker.put(("initialization", True))
        self.query_from_worker.put(("memid", agent.memory.self_memid))
        while True:
            self.perceive(agent)
            self.send_perception_updates(agent)
            self.handle_input_task(agent)
            self.task_step(agent)
            self.handle_master_query(agent)
            agent.count += 1

