from droidlet.interpreter.craftassist import MCInterpreter
from droidlet.interpreter.craftassist import swarm_tasks
from typing import Tuple, Any, Optional


class SwarmMCInterpreter(MCInterpreter):
    # self, speaker, logical_form_memid, agent_memory, memid=None, low_level_data=None):
    def __init__(self, speaker: str, logical_form_memid, agent_memory, memid=None, low_level_data=None):
        super(SwarmMCInterpreter, self).__init__(speaker, logical_form_memid, agent_memory, memid, low_level_data)
        # self.task_objects = {
        #     "move": tasks.Move,
        #     "undo": tasks.Undo,
        #     "build": tasks.Build,
        #     "destroy": tasks.Destroy,
        #     "spawn": tasks.Spawn,
        #     "fill": tasks.Fill,
        #     "dig": tasks.Dig,
        #     "dance": tasks.Dance,
        #     "point": tasks.Point,
        #     "dancemove": tasks.DanceMove,
        #     "get": tasks.Get,
        #     "drop": tasks.Drop,
        #     "control": ControlBlock,
        # }
        self.task_objects = super(SwarmMCInterpreter, self).task_objects
        # only override certain tasks : Move, build, destroy, dance, dancemove and dig supported for swarms here
        self.task_objects["move"] = swarm_tasks.SwarmMove
        self.task_objects["build"] = swarm_tasks.SwarmBuild
        self.task_objects["destroy"] = swarm_tasks.SwarmDestroy
        self.task_objects["dance"] = swarm_tasks.SwarmDance
        self.task_objects["dancemove"] = swarm_tasks.SwarmDanceMove
        self.task_objects["dig"] = swarm_tasks.SwarmDig
        self.task_objects["spawn"] = swarm_tasks.SwarmSpawn

    ### Override the handle stop and handle resume ###
    # TODO: need to test the following again in the new task format.!!!!
    def handle_stop(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if self.loop_data is not None:
            # TODO if we want to be able stop and resume old tasks, will need to store
            self.archived_loop_data = self.loop_data
            self.loop_data = None
        if hasattr(agent, "swarm_workers"):
            for i in range(agent.num_agents-1):
                agent.swarm_workers[i].query_from_master.put(("stop", None))
        if self.memory.task_stack_pause():
            return None, "Stopping.  What should I do next?", None
        else:
            return None, "I am not doing anything", None

    # FIXME this is needs updating...
    # TODO mark in memory it was resumed by command
    def handle_resume(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        self.finished = True
        if hasattr(agent, "swarm_workers"):
            for i in range(agent.num_agents-1):
                agent.swarm_workers[i].query_from_master.put(("resume", None))
        if self.memory.task_stack_resume():
            if self.archived_loop_data is not None:
                # TODO if we want to be able stop and resume old tasks, will need to store
                self.loop_data = self.archived_loop_data
                self.archived_loop_data = None
            return None, "resuming", None
        else:
            return None, "nothing to resume", None
