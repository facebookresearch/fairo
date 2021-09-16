from copy import deepcopy
import numpy as np

from droidlet.interpreter.task import Task
from droidlet.memory.memory_nodes import TaskNode
from droidlet.interpreter.craftassist import tasks
import pdb

# single agent task reference
TASK_MAP = {
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
            # "control": ControlBlock,
        }

def get_worker_idx_from_memid(swarm_workers_memid, memid):
    if memid not in swarm_workers_memid:
        return 0
    else:
        return (swarm_workers_memid.index(memid) + 1)

class BaseSwarmTask(Task):
    """
    Base Task Class for the swarm
    """
    memory_tag = "swarm_worker_{}"
    def __init__(self, agent, task_data={}, subcontrol='equal'):
        super().__init__(agent, task_data)
        # movement should be a Movement object from dance.py
        self.all_swarm_workers_memid = agent.swarm_workers_memid
        self.last_stepped_time = agent.memory.get_time()
        
        # self.swarm_worker_tasks = []
        self.get_task_agents(agent, task_data)
        self.distribute(task_data)
        TaskNode(agent.memory, self.memid).update_task(task=self)
    
    def get_task_agents(self, agent, task_data):
        self.task_agents_memid = task_data.get("task_agents_memid")
        if self.task_agents_memid is None:
            self.task_agents_memid = [agent.memory.self_memid] + agent.swarm_workers_memid
        self.num_agents = len(self.task_agents_memid)

    def distribute(self, task_data):
        """divide task to swarm workers
        """
        raise NotImplementedError
    
    def assign_to_worker(self, worker_memid, task_name, task_data):
        worker_idx = get_worker_idx_from_memid(self.all_swarm_workers_memid, worker_memid)
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
            self.assign_to_worker(self.task_agents_memid[i], "move", task_data)

class SwarmBuild(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
    
    def distribute(self, task_data):
        self.task_data = task_data
        block_list = task_data["blocks_list"]
        block_list.sort(key=lambda x: x[0][0])
        self.num_blocks = len(block_list)
        self.num_blocks_per_agent = np.array([self.num_blocks//self.num_agents] * self.num_agents)
        self.num_blocks_per_agent[-1] += self.num_blocks - self.num_blocks//self.num_agents * self.num_agents
        tmp_ind = 0
        for i in range(self.num_agents):
            tmp_task_data = deepcopy(task_data)
            tmp_task_data["blocks_list"] = block_list[tmp_ind: tmp_ind + self.num_blocks_per_agent[i]]
            tmp_blocks_array = np.array([(x, y, z, b, m) for ((x, y, z), (b, m)) in tmp_task_data["blocks_list"]])
            offset = np.min(tmp_blocks_array[:, :3], axis=0)
            # get offset to modify the origin
            tmp_task_data["origin"] += np.array(offset)
            tmp_ind += self.num_blocks_per_agent[i]
            self.assign_to_worker(self.task_agents_memid[i], "build", tmp_task_data)

class SwarmDestroy(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)

    def distribute(self, task_data):
        self.schematic = task_data["schematic"]
        self.schematic.sort(key=lambda x: x[0][0])
        self.num_blocks = len(self.schematic)
        self.num_blocks_per_agent = np.array([self.num_blocks//self.num_agents] * self.num_agents)
        self.num_blocks_per_agent[-1] += self.num_blocks - self.num_blocks//self.num_agents * self.num_agents
        tmp_ind = 0
        for i in range(self.num_agents):
            tmp_task_data = deepcopy(task_data)
            tmp_task_data["schematic"] = self.schematic[tmp_ind: tmp_ind + self.num_blocks_per_agent[i]]
            tmp_ind += self.num_blocks_per_agent[i]
            self.assign_to_worker(self.task_agents_memid[i], "destroy", tmp_task_data)

class SwarmDig(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        
    def distribute(self, task_data):
        self.origin = task_data["origin"]
        self.length = task_data["length"]
        self.width = task_data["width"]
        self.depth = task_data["depth"]
        mx, My, mz = self.origin
        Mx = mx + (self.width - 1)
        my = My - (self.depth - 1)
        Mz = mz + (self.length - 1)

        # can't divide work in the z direction, would create infeasible path for sub workers
        scheme = np.zeros([self.num_agents, 3], dtype=np.int32)
        scheme[:] = np.array([self.width // self.num_agents, self.depth, self.length])

        offset = np.zeros([self.num_agents], dtype = np.int32)
        offset[:] = self.width//self.num_agents
        
        offset[-1] += self.width - self.width// self.num_agents * self.num_agents
        scheme[-1, 0] += self.width - self.width// self.num_agents * self.num_agents

        tmp_m = np.array([mx, my, mz])
        for i in range(self.num_agents):
            tmp_task_data = deepcopy(task_data)
            tmp_task_data["origin"] = np.array([tmp_m[0], tmp_m[1] + scheme[i,1] - 1, tmp_m[2]])
            tmp_task_data['width'] = scheme[i, 0]
            tmp_task_data['depth'] = scheme[i, 1]
            tmp_task_data['length'] = scheme[i, 2]
            tmp_m[0] = tmp_m[0] + offset[i]
            if np.min(scheme[i]) <= 0:
                continue
            self.assign_to_worker(self.task_agents_memid[i], "dig", tmp_task_data)
