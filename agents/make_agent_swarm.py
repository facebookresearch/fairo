import logging

from agents.swarm_utils import get_memory_handlers_dict, get_swarm_interpreter, safe_object
from agents.swarm_worker_wrapper_process import SwarmWorkerProcessWrapper
from droidlet.dialog.swarm_dialogue_manager import SwarmDialogueManager


class SwarmMasterWrapper():
    def __init__(self, agent=None, worker_agents=[], opts=None) -> None:
        # init agent and type
        self.agent = agent
        self.agent_type = agent.__class__.__name__.lower()
        self.opts = opts

        # init workers
        self.num_workers = len(worker_agents) 
        self.init_controller()
        self.init_all_workers(worker_agents, self.opts)
        self.init_memory_handlers_dict()
        
        # task_step_filters is used to filter out tasks meant for worker agents
        self.agent.task_step_filters = ["worker_bot_{}".format(i+1) for i in range(self.num_workers)]
        self.agent.num_agents = self.num_workers + 1 # including main agent

    def init_all_workers(self, worker_agents, opts):
        # get task object mappings for interpreter. 
        # TODO: set these here for Put and get memory eventually
        task_object_mapping = self.agent.dialogue_manager.\
            dialogue_objects["interpreter"].task_objects
        
        # init Swarm workers as a list
        self.swarm_workers = []
        for i in range(self.num_workers):
            self.swarm_workers.append(SwarmWorkerProcessWrapper(task_object_mapping=task_object_mapping, worker_index = i+1, opts=opts))
        
        # init and store worker memids.
        self.agent.swarm_workers_memid = [None for i in range(self.num_workers)]
        self.swarm_workers_memid = self.agent.swarm_workers_memid

    def init_controller(self):
        dialogue_object_classes = self.agent.dialogue_manager.dialogue_objects
        # overwrite the interpreter and init dialogue manager for swarm that overwrites the get_last_m_chats
        dialogue_object_classes["interpreter"] = get_swarm_interpreter(self.base_agent)
        self.base_agent.dialogue_manager = SwarmDialogueManager(
                memory=self.base_agent.memory,
                dialogue_object_classes=dialogue_object_classes,
                opts=self.base_agent.opts,
                low_level_interpreter_data=self.base_agent.dialogue_manager.low_level_interpreter_data
            )

    def init_memory_handlers_dict(self):
        """
        Get all methods of memory from droidlet agent +
        craftassist / locobot
        map of {method_name: method}
        """
        # TODO: customize to different agent
        self.handle_query_dict = get_memory_handlers_dict(self.base_agent)

    def handle_worker_perception(self):
        """
        Get perception output from each worker and write to main agent's
        memory
        :return:
        """
        # one perception map for each worker
        tmp_perceptions = [{} for i in range(self.num_workers)]
        worker_eids = dict()
        # Iterate over all workers
        for i in range(self.num_workers):
            # while things still in worker's perception queue
            while not self.swarm_workers[i].perceptions.empty:
                eid, name, obj = self.swarm_workers[i].perceptions.get_nowait()
                tmp_perceptions[i][name] = obj
                worker_eids[i] = eid 
        
        # TODO: resolve conflicts 
                              
        # write perception info back to memory
        for i in range(self.num_workers):
            if i not in worker_eids.keys():
                continue # this shouldn't happen
            eid = worker_eids[i]
            if "pos" in tmp_perceptions[i].keys():
                mem = self.agent.memory.get_player_by_eid(eid)
                memid = mem.memid
                cmd = (
                    "UPDATE ReferenceObjects SET eid=?, x=?, y=?, z=? WHERE uuid=?"
                )
                self.agent.memory.db_write(cmd, eid, tmp_perceptions[i]["pos"].x, 
                                                tmp_perceptions[i]["pos"].y, 
                                                tmp_perceptions[i]["pos"].z, memid)

    def assign_new_tasks_to_workers(self):
        """
        get new tasks for each worker and add to the channel that they are
        listening on
        :return:
        """
        for i in range(self.num_workers):
            task_list = self.get_new_tasks(tag="worker_bot_{}".format(i+1))
            for new_task in task_list:
                self.swarm_workers[i].input_task_queue.put(new_task)

    def update_tasks_with_worker_data(self):
        """
        update task status with info sent from workers
        """
        for i in range(self.num_workers):
            # query_or_updates_from_worker:
            # worker sends its general query to master in the
            # queue. e.g. task updates sent to the master
            # query from worker -> task updates, init status, worker memid
            while not self.swarm_workers[i].query_or_updates_from_worker.empty():
                name, obj = self.swarm_workers[i].query_or_updates_from_worker.get_nowait()
                if name == "task_updates":
                    for (memid, cur_task_status) in obj:
                        # memid = task memid
                        # task status is a tuple of status, so cur_task_status = (prio, running, finished)
                        mem = self.agent.memory.get_mem_by_id(memid)
                        mem.get_update_status({"prio": cur_task_status[0], "running": cur_task_status[1]})
                        if cur_task_status[2]: # if marked as finished
                            mem.task.finished = True
                elif name == "initialization":
                    # signal indicating the worker initialization is finished
                    # the main loop of the master agent starts after all workers initialization is done
                    self.init_status[i] = True
                elif name == "memid":
                    # the master receives each worker's memid and stores them
                    self.swarm_workers_memid[i] = obj
    
    def handle_worker_memory_queries(self):
        """
        go over each worker's memory query and send response to the
        query in their receive queues.
        handles the workers' queries of the master agent's memory 
        self.swarm_workers[i].memory_query_from_worker: the queue where swarm worker
        i sends its memory queries to the master
        """
        for i in range(self.num_workers):
            # memory_query_from_worker: worker send its memory related query to the master through this queue
            while not self.swarm_workers[i].memory_query_from_worker.empty():
                query = self.swarm_workers[i].memory_query_from_worker.get_nowait()
                response = self.handle_memory_query(query)
                # memory_query_answer_from_master: worker receives the memory query response from master from the queue
                self.swarm_workers[i].memory_query_answer_from_master.put(response)

    def handle_memory_query(self, query):
        """
        handle one memory query at a time from a worker. Here:
        query = (query_id, query_name, query_args) where:
        query_id is unique id for each query, we need the id to be able to send the response back
        to the worker. query_name is the query function name. e.g. db_write, tag, etc and
        query_args contains arguments for the query
        """
        query_id = query[0]
        query_name = query[1]
        query_args = query[2:]
        if query_name in self.handle_query_dict.keys(): # function name right now, can be made better.
            to_return = self.handle_query_dict[query_name](*query_args)
        else:
            logging.info("The main agent cannot handle memory query: {} yet.".format(query))
            raise NotImplementedError
        to_return = safe_object(to_return) # create a picklable object or tuple of picklable objects
        return tuple([query_id, to_return])


    def start(self):
        # count forever unless the shutdown signal is given
        # start each worker process
        for swarm_worker in self.swarm_workers:
            swarm_worker.start() # this will set init status.

        self.init_status = [False] * (self.num_workers)
        while not self.agent._shutdown:
            try:
                # wait for all processes to get initialized first
                if all(self.init_status):
                    self.agent.step()
                self.handle_worker_perception()
                self.assign_new_tasks_to_workers()
                self.update_tasks_with_worker_data() # updates init status.
                self.handle_worker_memory_queries()
                                
            except Exception as e:
                self.agent.handle_exception(e)
        for swarm_worker in self.swarm_workers:
            swarm_worker.join()
