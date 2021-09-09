from copy import deepcopy
import numpy as np

from droidlet.interpreter.task import Task
from droidlet.memory.memory_nodes import TaskNode
from droidlet.interpreter.robot import tasks
import pdb

# single agent task reference
TASK_MAP = {
        "move": tasks.Move,
        "look": tasks.Look,
        "dance": tasks.Dance,
        "point": tasks.Point,
        "turn": tasks.Turn,
        "autograsp": tasks.AutoGrasp,
        # "control": ControlBlock,
        "get": tasks.Get,
        "drop": tasks.Drop,
        }

class BaseSwarmTask(Task):
    """
    Base Task Class for the swarm
    """
    memory_tag = "swarm_worker_{}"
    def __init__(self, agent, task_data={}, subcontrol='equal'):
        super().__init__(agent, task_data)
        # movement should be a Movement object from dance.py
        # assert hasattr(self.agent, 'swarm_workers')
        self.num_agents = self.agent.num_agents
        
        self.distribute(task_data)
        TaskNode(agent.memory, self.memid).update_task(task=self)
    

    def distribute(self, task_data):
        """divide task to swarm workers
        """
        raise NotImplementedError
    
    def assign_to_worker(self, worker_idx, task_name, task_data):
        if worker_idx == 0:
            TASK_MAP[task_name](self.agent, task_data)
        else:
            tmp_task = TASK_MAP[task_name](self.agent, task_data)
            self.agent.memory.tag(tmp_task.memid, self.memory_tag.format(worker_idx))

class SwarmMove(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
    
    def distribute(self, task_data):
        for i in range(self.num_agents):
            self.assign_to_worker(i, "move", task_data)