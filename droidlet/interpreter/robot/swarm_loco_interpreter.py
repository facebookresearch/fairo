from droidlet.interpreter.robot import LocoInterpreter
from droidlet.interpreter.robot import swarm_tasks
from typing import Tuple, Dict, Any, Optional

class SwarmLocoInterpreter(LocoInterpreter):
    def __init__(self, speaker: str, action_dict: Dict, low_level_data: Dict = None, **kwargs):
        super(LocoInterpreter, self).__init__(speaker, action_dict, low_level_data, **kwargs)
        self.task_objects = super(LocoInterpreter, self).task_objects
        # self.task_objects = {
        #     "move": tasks.Move,
        #     "look": tasks.Look,
        #     "dance": tasks.Dance,
        #     "point": tasks.Point,
        #     "turn": tasks.Turn,
        #     "autograsp": tasks.AutoGrasp,
        #     "control": ControlBlock,
        #     "get": tasks.Get,
        #     "drop": tasks.Drop,
        # }
        self.task_objects["move"] = swarm_tasks.SwarmMove

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
