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

TASK_MAP = TASK_MAP = {
        "move": tasks.Move,
        # "build": tasks.Build,
        # "destroy": tasks.Destroy,
        # "dig": tasks.Dig,       
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
    def __init__(self, opts, idx=0):
        self.agent_idx = idx
        self.task_stacks = dict()
        self.prio = dict()
        self.running = dict()
        self.pause = False
        super(CraftAssistSwarmWorker, self).__init__(opts)

    def init_event_handlers(self):
        pass

    def init_perception(self):
        """Initialize perception modules"""
        self.perception_modules = {}
        self.perception_modules["low_level"] = SwarmLowLevelMCPerception(self)

    # def init_memory(self):
    #     self.memory_send_queue = Queue()
    #     self.memory_receive_queue = Queue()

    def init_controller(self):
        """Initialize all controllers"""
        pass

    def perceive(self, force=False):
        for v in self.perception_modules.values():
            v.perceive(force=force)
    
    def task_step(self):
        queries = []
        task_updates = []
        finished_task_memids = []
        for memid, task in self.task_stacks.items():
            # ForkedPdb().set_trace()
            pre_task_status = (self.prio[memid], self.running[memid], task.finished)
            if self.prio[memid] == -1:
                if task.init_condition.check():
                    self.prio[memid] = 0
            if (not self.pause) and (self.prio[memid] >=0):
                if task.run_condition.check():
                    self.prio[memid] = 1
                    self.running[memid] = 1
                if task.stop_condition.check():
                    self.prio[memid] = 0
                    self.running[memid] = 0
            cur_task_status = (self.prio[memid], self.running[memid], task.finished)
            if (not self.pause) and (self.running[memid] >=1):
                tmp_query = task.step()
                # TODO: return queries if anything is missing
                if task.finished:
                    finished_task_memids.append(memid)
                    cur_task_status = (self.prio[memid], self.running[memid], task.finished)
                if tmp_query is not None:
                    queries.append((memid, tmp_query))
            if cur_task_status!= pre_task_status:
                task_updates.append((memid, cur_task_status))
        
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
        self.input_chats = Queue()
        self.perceptions = Queue()
        self.query_from_worker = Queue()
        self.query_from_master = Queue()

    def send_queries(self, queries):
        # TODO: send queries to master by pushing to self.query_from_worker
        pass

    def send_task_updates(self, task_updates):
        # TODO: send task updates to master by pushing to self.query_from_worker
        name = 'task_updates'
        self.query_from_worker.put((name, task_updates))

    def update_task(self, memid, info):
        # TODO: update task info based on information given by master agent
        pass

    def run(self):  
        agent = CraftAssistSwarmWorker(self.opts, self.idx)
        while True:
            # ForkedPdb().set_trace()
            worker_perception = agent.perceive()
            self.perceptions.put(worker_perception)
            
            flag = True
            while flag:
                try:
                    memid, info = self.query_from_master.get_nowait()
                    self.update_task(memid, info)
                except:
                    flag = False

            try:
                task_class_name, task_data, task_memid = self.input_tasks.get_nowait()
                if task_memid not in agent.task_stacks.keys():
                # TODO: implement stop and resume
                    agent.task_stacks[task_memid] = TASK_MAP[task_class_name](agent, task_data)
                    agent.prio[task_memid] = -1
                    agent.running[task_memid] = -1
            except:
                logging.debug("swarm worker {}: no new task received".format(self.idx))
                pass
            
            try:
                chat = self.input_chats.get_nowait()
                agent.send_chat(chat)
            except:
                pass

            queries, task_updates = agent.task_step()
            self.send_queries(queries)
            self.send_task_updates(task_updates)
            agent.count += 1