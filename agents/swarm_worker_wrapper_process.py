import logging
from multiprocessing import Process, Queue
from copy import deepcopy

from agents.swarm_utils import get_default_task_info
from agents.craftassist.craftassist_agent import CraftAssistAgent
from droidlet.memory.swarm_worker_memory import SwarmWorkerMemory
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception


class SwarmWorkerProcessWrapper(Process):
    def __init__(self, task_object_mapping={}, worker_index=0, opts=None) -> None:
        super().__init__()
        self.opts = opts
        self.worker_index = worker_index
        ### Queues for communicating with master ###
        # input_task_queue: master agent sends a task to the worker through the queue
        self.input_task_queue = Queue()

        # output_perception_info_queue: send perception information to the master through the queue
        self.output_perception_info_queue = Queue()
        
        # query_or_updates_from_worker: worker sends its general query + updates to master in the queue. e.g. task updates sent to the master
        self.query_or_updates_from_worker = Queue()

        # query_from_master: worker receives the query/commands from the master from the queue
        self.query_from_master = Queue()

        # memory_send_queue: worker send its memory related query to the master through this queue
        self.memory_query_from_worker = Queue()

        # memory_query_answer_from_master: worker receives the memory query response from master from the queue
        self.memory_query_answer_from_master = Queue()
        
        # Initialize worker's task object mapping and info for each task
        self.init_task_map_and_info(task_object_mapping)

    
    def init_task_map_and_info(self, task_object_mapping, task_info=None):
        self.task_object_mapping = deepcopy(task_object_mapping)
        # get default info for every task in task_object_mapping.
        self.task_object_info = get_default_task_info(task_object_mapping)  # populate args of tasks from default map else []
        
        # if additional task_info has been sent, overwrite task info with input map
        if task_info is not None:
            for key in task_info:
                self.task_object_info[key] = task_info[key]


    def init_worker(self, agent):
        self.agent_type = agent.__class__.__name__.lower()
        agent.agent_index = self.worker_index
        
        # swarm worker local task management
        # task_stack store current tasks
        # duplicate_tasks store duplicated task memid sent from the master
        # prio, running, pause stores the priority, running status, stop status of each task
        agent.task_stack = {}
        agent.duplicate_tasks = []
        agent.prio = {}
        agent.running = {}
        agent.pause = {}

        # queues for communicating with the master agent
        agent.memory_query_from_worker = self.memory_query_from_worker
        agent.memory_query_answer_from_master = self.memory_query_answer_from_master
        agent.query_or_updates_from_worker = self.query_or_updates_from_worker
        agent.query_from_master = self.query_from_master

        #### temporary for debug
        agent.perception_modules = dict()
        agent.perception_modules["low_level"] = SwarmLowLevelMCPerception(agent)
        #### end temporary for debug

        # memory
        # memory_query_from_worker: worker send its memory related query to the master through this queue
        # memory_query_answer_from_master: worker receives the memory query response from master from the queue
        agent.memory = SwarmWorkerMemory(memory_send_queue=self.memory_query_from_worker,
                                         memory_receive_queue=self.memory_query_answer_from_master,
                                         memory_tag="worker_bot_{}".format(agent.agent_index))
        # controller
        agent.disable_chat = True

    
    def perceive(self, agent, force=False):
        for v in agent.perception_modules.values():
            v.perceive(force=force)

    
    def get_task_data(self, task_data):
        if "task_data" in task_data:
            return task_data["task_data"]
        else:
            return task_data


    def check_task_info_completeness(self, task_name, task_data):
        """
        created this for sanity check
        reject the task if the full task information is incomplete from the master
        the function is necessary for multiprocessing
        """
        if task_name not in self.TASK_INFO.keys():
            logging.info("task {} received without checking arguments")
            return True
        for key in self.TASK_INFO[task_name.lower()]:
            if key not in task_data:
                return False
        return True
    
    def send_task_updates(self, task_updates):
        """send task updates to master by pushing to self.query_or_updates_from_worker
        task_updates = [(task_memid, (
                agent.prio[task_memid], agent.running[task_memid], agent.task_stack[task_memid].finished))]
        """
        if len(task_updates) > 0:
            name = 'task_updates'
            # query_or_updates_from_worker: worker send its general query to master in the queue.
            self.query_or_updates_from_worker.put((name, task_updates))

    
    def task_step(self, agent):
        # Set the priority of tasks and send update to master
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stack.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if agent.prio[memid] == -1:  # new task
                if task.init_condition.check():  # if init condition is true, set pri to be run
                    agent.prio[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status != pre_task_status:
                task_updates.append((memid, cur_task_status))
        # send task updates when there's a change in task status
        self.send_task_updates(task_updates)

        # Set running status and priority if prio > -1 and not paused, send updated to master
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stack.items():
            pre_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if (not agent.pause[memid]) and (agent.prio[memid] > -1):
                # if task.run_condition.check():  # can it be run ?
                agent.prio[memid] = 1
                agent.running[memid] = 1
                # if task.stop_condition.check():  # does it need to be stoppped ?
                #     agent.prio[memid] = 0
                #     agent.running[memid] = 0
            cur_task_status = (agent.prio[memid], agent.running[memid], task.finished)
            if cur_task_status != pre_task_status:
                task_updates.append((memid, cur_task_status))
        # send task updates once the task status is changed
        self.send_task_updates(task_updates)


        # Invokd task.step() if running >=1 and not paused, send updates to master
        task_updates = []
        finished_task_memids = []
        for memid, task in agent.task_stack.items():
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

        # For tasks finished, delete from task_stack, priority and running dicts
        for memid in finished_task_memids:
            del agent.task_stack[memid]
            del agent.prio[memid]
            del agent.running[memid]
        return task_updates


    def handle_input_task(self, agent):
        """
        go over input_tasks queue, pick task_class_name, task_data, task_memid
        and create and add tasks to agent.task_stacks
        :param agent:
        :return:
        """
        while not self.input_task_queue.empty():
            task_class_name, task_data, task_memid = self.input_task_queue.get_nowait()
            if task_class_name not in self.TASK_MAP.keys():
                logging.info("task cannot be handled by this worker right now.")
                continue
            # Handle new task
            if task_memid is None or (
                    (task_memid not in agent.task_stack.keys()) and (task_memid not in agent.duplicate_tasks)):
                # if it is a new task, check the info completeness and then create it
                task_data = self.get_task_data(task_data)
                if self.check_task_info_completeness(task_class_name, task_data):
                    new_task = self.TASK_MAP[task_class_name](agent, task_data)
                    if task_memid is None:
                        task_memid = new_task.memid
                    else:
                        # add this to a duplicate task.
                        agent.duplicate_tasks.append(new_task.memid)
                    # can send updates back to main agent to mark as finished
                    agent.task_stack[task_memid] = new_task
                    agent.prio[task_memid] = -1
                    agent.running[task_memid] = -1
                    agent.pause[task_memid] = False
            elif task_memid in agent.task_stack.keys():
                # if it is an existing task, update master with the existing task status
                self.send_task_updates([(task_memid, (
                agent.prio[task_memid], agent.running[task_memid], agent.task_stack[task_memid].finished))])
            elif task_memid in agent.duplicate_tasks:
                # if it is a ghost task(ie: duplicate task), update master about task status so that it won't be
                # sent to the worker again
                self.send_task_updates([(task_memid, (0, 0, True))])

    def handle_master_query(self, agent):
        # query_from_master, worker receives the query/commands from the master from the queue
        while not self.query_from_master.empty():
            query_name, query_data = self.query_from_master.get_nowait()
            # Hand;ing stop and resume as unique cases right now...
            if query_name == "stop":
                for memid, task in agent.task_stack.items():
                    if not task.finished:
                        agent.pause[memid] = True
            elif query_name == "resume":
                for memid, task in agent.task_stack.items():
                    agent.pause[memid] = False
            else:
                logging.info("Query not currently handled by the worker : {}".format(query_name))
                raise NotImplementedError

    def run(self):
        # this is what happens when the process is started -> when
        # process.start() is called.
        self.opts.name = "worker_bot_" + str(self.worker_index)
        agent = CraftAssistAgent(self.opts)
        # Init the worker with CA agent and let master know of memid and init completion
        self.init_worker(agent)
        self.query_or_updates_from_worker.put(("initialization", True))
        # TODO: look into why we need agent's memory ?
        self.query_or_updates_from_worker.put(("memid", agent.memory.self_memid))

        while True:
            self.perceive(agent)
            # self.send_perception_updates(agent) -- skip, send no updates back right now
            self.handle_input_task(agent)
            self.task_step(agent)
            self.handle_master_query(agent)
            agent.count += 1