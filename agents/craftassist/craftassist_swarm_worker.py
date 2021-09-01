"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception
from agents.craftassist.craftassist_agent import CraftAssistAgent
from multiprocessing import Process, Queue
from droidlet.interpreter.craftassist import tasks
from droidlet.interpreter.task import ControlBlock
import logging
import pdb
import sys
from droidlet.lowlevel.minecraft.mc_util import MCTime
from droidlet.memory.craftassist.swarm_worker_memory import SwarmWorkerMemory

TASK_MAP =  {
        "move": tasks.Move,
        "build": tasks.Build,
        # "destroy": tasks.Destroy,
        # "dig": tasks.Dig,       
    }

TASK_INFO = {
    "move": ["target"],
    "build": []
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

class CraftAssistSwarmWorker(CraftAssistAgent):
    def __init__(self, opts, idx, memory_send_queue, memory_receive_queue, query_from_worker):
        self.agent_idx = idx
        self.task_stacks = dict()
        self.task_ghosts = []
        self.prio = dict()
        self.running = dict()
        self.pause = False
        self.memory_send_queue = memory_send_queue
        self.memory_receive_queue = memory_receive_queue
        self.query_from_worker = query_from_worker
        super(CraftAssistSwarmWorker, self).__init__(opts)

    def init_event_handlers(self):
        pass

    def init_perception(self):
        """Initialize perception modules"""
        self.perception_modules = {}
        self.perception_modules["low_level"] = SwarmLowLevelMCPerception(self)

    def init_memory(self):
        self.memory = SwarmWorkerMemory(agent_time=MCTime(self.get_world_time),
                                        memory_send_queue=self.memory_send_queue,
                                        memory_receive_queue=self.memory_receive_queue,
                                        memory_tag="swarm_worker_{}".format(self.agent_idx))

    def init_controller(self):
        """Initialize all controllers"""
        pass

    def perceive(self, force=False):
        for v in self.perception_modules.values():
            v.perceive(force=force)
    
    def check_task_info(self, task_name, task_data):
        if task_name not in TASK_INFO.keys():
            logging.info("task {} received without checking arguments")
            return True
        for key in TASK_INFO[task_name.lower()]:
            if key not in task_data:
                return False
        return True

    def send_task_updates(self, task_updates):
        # TODO: send task updates to master by pushing to self.query_from_worker
        if len(task_updates)>0:
            name = 'task_updates'
            self.query_from_worker.put((name, task_updates))

    def task_step(self):
        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in self.task_stacks.items():
            pre_task_status = (self.prio[memid], self.running[memid], task.finished)
            if self.prio[memid] == -1:
                if task.init_condition.check():
                    self.prio[memid] = 0
            cur_task_status = (self.prio[memid], self.running[memid], task.finished)
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in self.task_stacks.items():
            pre_task_status = (self.prio[memid], self.running[memid], task.finished)
            if (not self.pause) and (self.prio[memid] >=0):
                if task.run_condition.check():
                    self.prio[memid] = 1
                    self.running[memid] = 1
                if task.stop_condition.check():
                    self.prio[memid] = 0
                    self.running[memid] = 0
            cur_task_status = (self.prio[memid], self.running[memid], task.finished)
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        self.send_task_updates(task_updates)

        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in self.task_stacks.items():
            pre_task_status = (self.prio[memid], self.running[memid], task.finished)    
            if (not self.pause) and (self.running[memid] >=1):
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
            del self.task_stacks[memid]
            del self.prio[memid]
            del self.running[memid]
        return queries, task_updates

class CraftAssistSwarmWorker_Wrapper(Process):
    def __init__(self, opts, idx=0):
        super().__init__()
        self.opts = opts
        self.idx = idx
        self.input_tasks = Queue()
        self.perceptions = Queue()
        self.query_from_worker = Queue()
        self.query_from_master = Queue()
        self.memory_send_queue = Queue()
        self.memory_receive_queue = Queue()

    def send_queries(self, queries):
        # TODO: send queries to master by pushing to self.query_from_worker
        pass
    
    def update_task(self, memid, info):
        # TODO: update task info based on information given by master agent
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
                # TODO: implement stop and resume
                    if agent.check_task_info(task_class_name, task_data):
                        new_task = TASK_MAP[task_class_name](agent, task_data)
                        if task_memid is None:
                            task_memid = new_task.memid
                        else:
                            agent.task_ghosts.append(new_task.memid)
                            # can send updates back to main agent to mark as finished
                        agent.task_stacks[task_memid] = new_task
                        agent.prio[task_memid] = -1
                        agent.running[task_memid] = -1
                elif task_memid in agent.task_stacks.keys():
                    self.send_task_updates([(task_memid, (agent.prio[task_memid], agent.running[task_memid], agent.task_stacks[task_memid].finished))])
                elif task_memid in agent.task_ghosts:
                    self.send_task_updates([(task_memid, (0, 0, True))])

    def run(self):
        agent = CraftAssistSwarmWorker(self.opts, self.idx, memory_send_queue=self.memory_send_queue, memory_receive_queue=self.memory_receive_queue, query_from_worker=self.query_from_worker)
        self.query_from_worker.put(("initialization", True))
        while True:
            worker_perception = agent.perceive()
            self.perceptions.put(worker_perception)
            
            self.handle_input_task(agent)
            queries, _ = agent.task_step()
            self.send_queries(queries)
            agent.count += 1