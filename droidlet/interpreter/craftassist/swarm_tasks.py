from copy import deepcopy
import logging
import numpy as np
import time

from random import randint

from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    PASSABLE_BLOCKS,
    BUILD_BLOCK_REPLACE_MAP,
    BUILD_IGNORE_BLOCKS,
    BUILD_INTERCHANGEABLE_PAIRS,
)
from droidlet.base_util import npy_to_blocks_list, blocks_list_to_npy, MOBS_BY_ID
from droidlet.perception.craftassist import search
from droidlet.perception.craftassist.heuristic_perception import ground_height
from droidlet.lowlevel.minecraft.mc_util import to_block_pos, manhat_dist, strip_idmeta

from droidlet.interpreter.task import BaseMovementTask, Task
from droidlet.memory.memory_nodes import TaskNode, TripleNode
from droidlet.memory.craftassist.mc_memory_nodes import MobNode
from droidlet.interpreter.craftassist import tasks, dance
import pdb

# single agent task reference
# self.task_objects = {
#             "move": tasks.Move,
#             "undo": tasks.Undo,
#             "build": tasks.Build,
#             "destroy": tasks.Destroy,
#             "spawn": tasks.Spawn,
#             "fill": tasks.Fill,
#             "dig": tasks.Dig,
#             "dance": tasks.Dance,
#             "point": tasks.Point,
#             "dancemove": tasks.DanceMove,
#             "get": tasks.Get,
#             "drop": tasks.Drop,
#             "control": ControlBlock,
#         }

class BaseSwarmTask(Task):
    """
    Base Task Class for the swarm
    """

    def __init__(self, agent, task_data={}, subcontrol='equal'):
        super().__init__(agent, task_data)
        # movement should be a Movement object from dance.py
        assert hasattr(self.agent, 'swarm_workers')
        self.num_agents = self.agent.num_agents
        self.last_stepped_time = agent.memory.get_time()
        
        # self.swarm_worker_tasks = []
        self.distribute(task_data)
        TaskNode(agent.memory, self.memid).update_task(task=self)
    

    def distribute(self, task_data):
        """divide task to swarm workers
        examples::

        >>> self.swarm_worker_tasks = []
        >>> build_task = Build(
                worker_agent,
                {
                    "blocks_list": destroy_schm,
                    "origin": origin,
                    "force": True,
                    "verbose": False,
                    "embed": True,
                    "dig_message": self.dig_message,
                    "is_destroy_schm": not self.dig_message,
                    "DIG_REACH": self.DIG_REACH,
                },
            )
        >>> self.add_child_task(build_task)
        >>> self.swarm_worker_tasks.append(build_task)
        """
        raise NotImplementedError

# TODO: add task memid tracking and check if subtask is finished

class SwarmDance(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        # movement should be a Movement object from dance.py
        self.movement = task_data.get("movement")
    
    def distribute(self, task_data):
        self.swarm_worker_tasks = []
        for i in range(self.num_agents):
            swarm_worker = self.agent.swarm_workers[i]
            # cannot directly append the sub tasks to a list, db can't pickle.dump()
            tasks.Dance(agent=swarm_worker, task_data=task_data)

class SwarmDanceMove(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
        self.relative_yaw = task_data.get("relative_yaw")
        self.relative_pitch = task_data.get("relative_pitch")

        # look_turn is (yaw, pitch).  pitch = 0 is head flat
        self.head_yaw_pitch = task_data.get("head_yaw_pitch")
        self.head_xyz = task_data.get("head_xyz")

        self.translate = task_data.get("translate")
    
    def distribute(self, task_data):
        self.swarm_worker_tasks = []
        for i in range(self.num_agents):
            swarm_worker = self.agent.swarm_workers[i]
            tasks.Dance(agent=swarm_worker, task_data=task_data)


class SwarmMove(BaseSwarmTask):
    def __init__(self, agent, task_data):
        super().__init__(agent, task_data)
    
    def distribute(self, task_data):
        for i in range(self.num_agents):
            swarm_worker = self.agent.swarm_workers[i]
            tasks.Move(agent=swarm_worker, task_data=task_data)

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
            swarm_worker = self.agent.swarm_workers[i]
            tmp_task_data["blocks_list"] = block_list[tmp_ind: tmp_ind + self.num_blocks_per_agent[i]]
            tmp_blocks_array = np.array([(x, y, z, b, m) for ((x, y, z), (b, m)) in tmp_task_data["blocks_list"]])
            offset = np.min(tmp_blocks_array[:, :3], axis=0)
            # get offset to modify the origin
            tmp_task_data["origin"] += np.array(offset)
            tmp_ind += self.num_blocks_per_agent[i]
            tasks.Build(agent=swarm_worker, task_data=tmp_task_data)

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
            swarm_worker = self.agent.swarm_workers[i]
            tmp_task_data["schematic"] = self.schematic[tmp_ind: tmp_ind + self.num_blocks_per_agent[i]]
            tmp_ind += self.num_blocks_per_agent[i]
            tasks.Destroy(agent=swarm_worker, task_data=tmp_task_data)

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
            swarm_worker = self.agent.swarm_workers[i]
            tmp_task_data = deepcopy(task_data)
            tmp_task_data["origin"] = np.array([tmp_m[0], tmp_m[1] + scheme[i,1] - 1, tmp_m[2]])
            tmp_task_data['width'] = scheme[i, 0]
            tmp_task_data['depth'] = scheme[i, 1]
            tmp_task_data['length'] = scheme[i, 2]
            tmp_m[0] = tmp_m[0] + offset[i]
            if np.min(scheme[i]) <= 0:
                continue
            tasks.Dig(agent=swarm_worker, task_data=tmp_task_data)
