from droidlet.interpreter.craftassist import MCInterpreter
from droidlet.interpreter.craftassist import swarm_tasks
from typing import Tuple, Dict, Any, Optional

class SwarmMCInterpreter(MCInterpreter):
    def __init__(self, speaker: str, action_dict: Dict, low_level_data: Dict = None, **kwargs):
        super(SwarmMCInterpreter, self).__init__(speaker, action_dict, low_level_data, **kwargs)
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
        self.task_objects["move"] = swarm_tasks.SwarmMove
        self.task_objects["build"] = swarm_tasks.SwarmBuild
        self.task_objects["destroy"] = swarm_tasks.SwarmDestroy
        self.task_objects["dig"] = swarm_tasks.SwarmDig

    def handle_stop(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        assert hasattr(agent, 'swarm_workers')
        self.finished = True
        for i in range(agent.num_agents):
            swarm_worker = agent.swarm_workers[i]
            super().handle_stop(swarm_worker, speaker, d)

    # FIXME this is needs updating...
    # TODO mark in memory it was resumed by command
    def handle_resume(self, agent, speaker, d) -> Tuple[Optional[str], Any]:
        assert hasattr(agent, 'swarm_workers')
        self.finished = True
        for i in range(agent.num_agents):
            swarm_worker = agent.swarm_workers[i]
            super().handle_resume(swarm_worker, speaker, d)
