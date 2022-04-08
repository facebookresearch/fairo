
from agents.craftassist.craftassist_agent import CraftAssistAgent
import logging
from agents.swarm_configs import get_default_task_info
from multiprocessing import Process, Queue
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception
from droidlet.memory.swarm_worker_memory import SwarmWorkerMemory
from copy import deepcopy
import pdb, sys

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



class SwarmWorkerWrapper(Process):
    def __init__(self, opts, task_map, disable_perception_modules, idx=0) -> None:
        super().__init__()
        self.opts = opts
        self.idx = idx

        # input_tasks: master agent sent worker task to the worker through the queue
        self.input_tasks = Queue()

        # perceptions: might be removed later
        # perceptions: send perception information to the master through the queue
        self.perceptions = Queue()

        # queues for communicating with master

        # query_from_worker: worker send its general query to master in the queue. e.g. task updates sent to the master
        self.query_from_worker = Queue()

        # query_from_master, worker receive the query/commands from the master from the queue
        self.query_from_master = Queue()

        # memory_send_queue: worker send its memory related query to the master through this queue
        self.memory_send_queue = Queue()

        # memory_receive_queue: worker receives the memory query response from master from the queue
        self.memory_receive_queue = Queue()

        self.init_task_map(task_map)
        self.disable_perception_modules = disable_perception_modules

    def init_task_map(self, task_map, task_info=None):
        self.TASK_MAP = deepcopy(task_map)
        self.TASK_INFO = get_default_task_info(task_map)  # populate args of tasks from default map else []
        # overwrite task info with input
        if task_info is not None:
            for key in task_info:
                self.TASK_INFO[key] = task_info[key]

    def init_worker(self, agent):
        self.agent_type = agent.__class__.__name__.lower()

        agent.agent_idx = self.idx

        # swarm worker local task management
        # task_stacks store current tasks
        # task_ghosts store duplicated task memid sent from the master
        # prio, running, pause stores the priority, running status, stop status of each task
        agent.task_stacks = dict()
        agent.task_ghosts = []
        agent.prio = dict()
        agent.running = dict()
        agent.pause = dict()

        # queues for communicating with the master agent
        agent.memory_send_queue = self.memory_send_queue
        agent.memory_receive_queue = self.memory_receive_queue
        agent.query_from_worker = self.query_from_worker

        # disable perception modules
        for module_key in self.disable_perception_modules:
            del agent.perception_modules[module_key]

        #### temporary for debug
        agent.perception_modules = dict()
        agent.perception_modules["low_level"] = SwarmLowLevelMCPerception(agent)
        #### end temporary for debug

        # memory
        # memory_send_queue: worker send its memory related query to the master through this queue
        # memory_receive_queue: worker receives the memory query response from master from the queue
        agent.memory = SwarmWorkerMemory(memory_send_queue=self.memory_send_queue,
                                         memory_receive_queue=self.memory_receive_queue,
                                         memory_tag="swarm_worker_{}".format(agent.agent_idx))
        # controller
        agent.disable_chat = True

    def check_task_info(self, task_name, task_data):
        """
        create for sanity checking
        reject the task if the full task information is incomplete from the master
        the function is necessary because of the multiprocessing
        """
        if task_name not in self.TASK_INFO.keys():
            logging.info("task {} received without checking arguments")
            return True
        for key in self.TASK_INFO[task_name.lower()]:
            if key not in task_data:
                return False
        return True

    def preprocess_data(self, task_name, task_data):
        if "task_data" in task_data:
            return task_data["task_data"]
        else:
            return task_data

    def send_task_updates(self, task_updates):
        """send task updates to master by pushing to self.query_from_worker
        """
        if len(task_updates) > 0:
            name = 'task_updates'
            # query_from_worker: worker send its general query to master in the queue.
            self.query_from_worker.put((name, task_updates))

    def handle_input_task(self, agent):
        """
        go over input_tasks queue, pick task_class_name, task_data, task_memid
        and create and add tasks to agent.task_stacks
        :param agent:
        :return:
        """
        while not self.input_tasks.empty():
            task_class_name, task_data, task_memid = self.input_tasks.get_nowait()
            if task_class_name not in self.TASK_MAP.keys():
                logging.info("task cannot be handled by this worker right now.")
                continue
            if task_memid is None or (
                    (task_memid not in agent.task_stacks.keys()) and (task_memid not in agent.task_ghosts)):
                # if it is a new task, check the info completeness and then create it
                task_data = self.preprocess_data(task_class_name, task_data)
                if self.check_task_info(task_class_name, task_data):
                    new_task = self.TASK_MAP[task_class_name](agent, task_data)
                    if task_memid is None:
                        task_memid = new_task.memid
                    else:
                        agent.task_ghosts.append(new_task.memid)
                        # can send updates back to main agent to mark as finished
                    agent.task_stacks[task_memid] = new_task
                    agent.prio[task_memid] = -1
                    agent.running[task_memid] = -1
                    agent.pause[task_memid] = False
            elif task_memid in agent.task_stacks.keys():
                # if it is an existing task, update the master with the existing task status
                self.send_task_updates([(task_memid, (
                agent.prio[task_memid], agent.running[task_memid], agent.task_stacks[task_memid].finished))])
            elif task_memid in agent.task_ghosts:
                # if it is a ghost task(ie: duplicate task), update master about task status so that it won't be
                # sent to the worker again
                self.send_task_updates([(task_memid, (0, 0, True))])

    def handle_master_query(self, agent):
        # query_from_master, worker receives the query/commands from the master from the queue
        while not self.query_from_master.empty():
            query_name, query_data = self.query_from_master.get_nowait()
            if query_name == "stop":
                for memid, task in agent.task_stacks.items():
                    if not task.finished:
                        agent.pause[memid] = True
            elif query_name == "resume":
                for memid, task in agent.task_stacks.items():
                    agent.pause[memid] = False
            else:
                logging.info("Query not handled: {}".format(query_name))
                raise NotImplementedError

    # TOFIX -->
    def send_perception_updates(self, agent):
        pass

    def perceive(self, agent, force=False):
        for v in agent.perception_modules.values():
            v.perceive(force=force)

    def task_step(self, agent):
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if agent.prio[memid] == -1:  # new task
                if task.init_condition.check():  # if init condition is true, set pri to be run
                    agent.prio[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status != pre_task_status:
                task_updates.append((memid, cur_task_status))
        # send task updates when there's a change in task status
        self.send_task_updates(task_updates)

        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if (not agent.pause[memid]) and (agent.prio[memid] >= 0):
                if task.run_condition.check():  # can it be run ?
                    agent.prio[memid] = 1
                    agent.running[memid] = 1
                if task.stop_condition.check():  # does it need to be stoppped ?
                    agent.prio[memid] = 0
                    agent.running[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status != pre_task_status:
                task_updates.append((memid, cur_task_status))
        # send task updates once the task status is changed
        self.send_task_updates(task_updates)

        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stacks.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if (not agent.pause[memid]) and (agent.running[memid] >= 1):
                task.step()
                if task.finished:
                    finished_task_memids.append(memid)
                    cur_task_status = (0, 0, task.finished)
            if cur_task_status != pre_task_status:
                task_updates.append((memid, cur_task_status))
        # send task updates once the task status is changed
        self.send_task_updates(task_updates)

        for memid in finished_task_memids:
            del agent.task_stacks[memid]
            del agent.prio[memid]
            del agent.running[memid]
        return task_updates

    def run(self):
        # this is what happens when the process is started -> when
        # process.start() is called.
        agent = CraftAssistAgent(self.opts)
        # init the worker and let master know of initialization and memid
        self.init_worker(agent)
        self.query_from_worker.put(("initialization", True))
        self.query_from_worker.put(("memid", agent.memory.self_memid))
        while True:
            self.perceive(agent)
            self.send_perception_updates(agent)
            self.handle_input_task(agent)
            self.task_step(agent)
            self.handle_master_query(agent)
            agent.count += 1

