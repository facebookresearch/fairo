import numpy as np
from base_agent.task import Task

class Move(Task):
    def __init__(self, agent, task_data):
        super(Move, self).__init__(agent)
        self.target = task_data["target"]
    
    def step(self, agent):
        super().step(agent)
        if self.finished:
            return
        agent.move(self.target[0], self.target[1], self.target[2])
        self.finished = True


class Grab(Task):
    def __init__(self, agent, task_data):
        super(Grab, self).__init__(agent)
        self.target_eid = task_data["target_eid"]

    def step(self, agent):
        super().step(agent)
        if self.finished:
            return

        if len(agent.world.get_bots(eid=self.target_eid)) > 0:
            agent.catch(self.target_eid)
        else:
            self.finished = True



class Catch(Task):
    def __init__(self, agent, task_data):
        super(Catch, self).__init__(agent)
        self.target_memid = task_data["target_memid"]
    
    def step(self, agent):
        super().step(agent)
        if self.finished:
            return

        #retrieve target info from memory:
        target_mem = agent.memory.get_mem_by_id(self.target_memid)
                    
        # first get close to the target, one block at a time
        tx, ty, tz = target_mem.get_pos()
        x, y, z = agent.get_pos()
        if np.linalg.norm(np.subtract((x, y, z), (tx, ty, tz))) > 0.:
            if x != tx:
                x += 1 if x - tx < 0 else -1
            else:
                y += 1 if y - ty < 0 else -1
            move_task = Move(agent, {"target": (x, y, z)})
            agent.memory.add_tick()
            self.add_child_task(move_task, agent)
            return

        # once target is within reach, catch it!
        grab_task = Grab(agent, {"target_eid": target_mem.eid})
        agent.memory.add_tick()
        self.add_child_task(grab_task, agent)
        self.finished = True

