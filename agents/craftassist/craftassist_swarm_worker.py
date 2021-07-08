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

TASK_MAP = TASK_MAP = {
        "move": tasks.Move,
        "undo": tasks.Undo,
        "build": tasks.Build,
        "destroy": tasks.Destroy,
        "spawn": tasks.Spawn,
        "fill": tasks.Fill,
        "dig": tasks.Dig,
        "dance": tasks.Dance,
        "point": tasks.Point,
        "dancemove": tasks.DanceMove,
        "get": tasks.Get,
        "drop": tasks.Drop,
        "control": ControlBlock,
    }

class CraftAssistSwarmWorker(CraftAssistAgent):
    def __init__(self, opts, idx=0):
        self.agent_idx = idx
        super(CraftAssistSwarmWorker, self).__init__(opts)

    def init_event_handlers(self):
        pass

    def init_perception(self):
        """Initialize perception modules"""
        self.perception_modules = {}
        self.perception_modules["low_level"] = SwarmLowLevelMCPerception(self)

    def init_controller(self):
        """Initialize all controllers"""
        pass

    def perceive(self, force=False):
        # disable chat modules for worker perception
        # super().perceive()
        for v in self.perception_modules.values():
            v.perceive(force=force)
    
    def step(self):
        self.perceive()
        self.task_step()
        self.count += 1

class CraftAssistSwarmWorker_Wrapper(Process):
    def __init__(self, opts, idx=0):
        super().__init__()
        self.opts = opts
        self.idx = idx
        self.input_tasks = Queue()
        self.input_chats = Queue()

    def run(self):
        agent = CraftAssistSwarmWorker(self.opts, self.idx)
        while True:
            agent.step()
            try:
                task_class_name, task_data = self.input_tasks.get_nowait()
                # TODO: implement stop and resume
                TASK_MAP[task_class_name](agent, task_data)
            except:
                logging.debug("No new task for swarm worker")
                pass

            try:
                chat = self.input_chats.get_nowait()
                agent.send_chat(chat)
            except:
                pass
