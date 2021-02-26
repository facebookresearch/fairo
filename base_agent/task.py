"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# the default init_condition is Never.  in current interpreter,
# a *top-level* task from a command with a Never condition is forcibly activated
# this is because we don't expect a top-level Event or Action command with a never condition
# this should be revisited when we have methods for changing conditions
# and wish to handle things like "do x; wait, don't do it yet" without giving
# a condition for the event or action to start (but then later on explaining when the
# event or action should occur)
from condition import NeverCondition, AlwaysCondition, TaskStatusCondition, NotCondition
from memory_nodes import TaskNode, LocationNode, TripleNode

# FIXME TODO store conditions in memory (new table)
# TaskNode method to update a tasks conditions
# dsl/put_memory for commands to do so
def get_default_conditions(task_data, agent, task):
    init_condition = task_data.get("init_condition", AlwaysCondition(None))

    run_condition = task_data.get("run_condition")
    stop_condition = task_data.get("stop_condition")
    if stop_condition is None:
        if run_condition is None:
            stop_condition = NeverCondition(None)
            run_condition = AlwaysCondition(None)
        else:
            stop_condition = NotCondition(run_condition)
    elif run_condition is None:
        run_condition = NotCondition(stop_condition)

    remove_condition = task_data.get("remove_condition", TaskStatusCondition(agent, task.memid))
    return init_condition, stop_condition, run_condition, remove_condition


# put a counter and a max_count so can't get stuck?
class Task(object):
    """This class represents a Task, the exact implementation of which 
    will depend on the framework and environment. A task can be placed on a 
    task stack, and represents a unit (which in itself can contain a sequence of s
    smaller subtasks).

    Attributes:
        memid (string): Memory id of the task in agent's memory
        interrupted (bool): A flag indicating whetherr the task has been interrupted
        finished (bool): A flag indicating whether the task finished 
        name (string): Name of the task
        undone (bool): A flag indicating whether the task was undone / reverted
        last_stepped_time (int): Timestamp of last step through the task
        throttling_tick (int): The threshold beyond which the task will be throttled
        stop_condition (Condition): The condition on which the task will be stopped (by default,
                        this is NeverCondition)
    Examples::
        >>> Task()
    """

    def __init__(self, agent, task_data={}):
        self.agent = agent
        self.run_count = 0
        self.interrupted = False
        self.finished = False
        self.name = None
        self.undone = False
        self.last_stepped_time = None
        self.prio = -1
        self.running = 0
        self.memid = TaskNode.create(self.agent.memory, self)
        # TODO put these in memory in a new table?
        # TODO methods for safely changing these
        i, s, ru, re = get_default_conditions(task_data, agent, self)
        self.init_condition = i
        self.stop_condition = s
        self.run_condition = ru
        self.remove_condition = re
        TripleNode.create(
            self.agent.memory,
            subj=self.memid,
            pred_text="has_name",
            obj_text=self.__class__.__name__.lower(),
        )
        TaskNode(agent.memory, self.memid).update_task(task=self)
        # TODO if this is a command, put a chat_effect triple

    @staticmethod
    def step_wrapper(stepfn):
        def modified_step(self):
            if self.remove_condition.check():
                self.finished = True
            if self.finished:
                TaskNode(self.agent.memory, self.memid).get_update_status(
                    {"prio": -2, "finished": True}
                )
                return
            query = {
                "base_table": "Tasks",
                "base_range": {"minprio": 0.5},
                "triples": [{"pred_text": "_has_parent_task", "obj": self.memid}],
            }
            child_task_mems = self.agent.memory.basic_search(query)
            if child_task_mems:  # this task has active children, step them
                return
            r = stepfn(self)
            TaskNode(self.agent.memory, self.memid).update_task(task=self)
            return

        return modified_step

    def step(self):
        """The actual execution of a single step of the task is defined here."""
        pass

    # FIXME remove all this its dead now...
    def interrupt(self):
        """Interrupt the task and set the flag"""
        self.interrupted = True

    def check_finished(self):
        """Check if the task has marked itself finished
        
        Returns:
            bool: If the task has finished
        """
        if self.finished:
            return self.finished

    def add_child_task(self, t):
        TaskNode(self.agent.memory, self.memid).add_child_task(t)

    def __repr__(self):
        return str(type(self))


# if you want a list of tasks, have to enclose in a control block
# FIXME/TODO: name any nonpicklable attributes in the object
class ControlBlock(Task):
    """Container for task control
    

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
            task_data["new_tasks"] is either a list of Tasks, or a callable
                if it is a callable, when called it returns (Task, sequence_finished) or (None, True)
                when it returns None, this ControlBlock is finished.
                to make an infinite loop, the callable needs to keep returning Tasks; 
                this ControlBlocks loop counter is incremented when sequence_finished is True
                and left unchanged otherwise
            task_data["loop"] should be set to True if a list of Tasks is input,
                and the list should be looped.  default (unset) is False.  If the
                new_tasks_fn is a callable, this will be ignored
    """

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data=task_data)
        self.loop = task_data.get("loop", False)
        self.setup_tasks(task_data.get("new_tasks"))
        TaskNode(self.agent.memory, self.memid).update_task(task=self)

    def setup_tasks(self, tasks):
        # if tasks is a list, converts it into a callable
        # if its a callable, just adds it
        if callable(tasks):
            self.tasks_fn = tasks
        else:
            assert type(tasks) is list
            self.task_list = tasks
            for task in self.task_list:
                # WARNING !! the control block is going to forcibly init
                # its children.
                cdict = {"init_condition": NeverCondition(None)}
                TaskNode(self.agent.memory, task.memid).update_condition(cdict)
            self.task_list_idx = 0

            def fn():
                sequence_finished = False
                if self.task_list_idx >= len(self.task_list):
                    if self.loop:
                        self.task_list_idx = 0
                        # increment loop counter
                        sequence_finished = True
                    else:
                        # not supposed to be looping this:
                        assert self.run_count <= 1
                        return None, True
                task = self.task_list[self.task_list_idx]
                self.task_list_idx += 1
                return task, sequence_finished

            self.tasks_fn = fn

    @Task.step_wrapper
    def step(self):
        t, update_run_count = self.tasks_fn()
        import ipdb

        ipdb.set_trace()
        if update_run_count:
            self.run_count += 1
        if t is not None:
            self.add_child_task(t)
        else:
            self.finished = True


class BaseMovementTask(Task):
    """ a Task that changes the location of the agent

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
            task_data should have a key "target" with a target location

    Attributes: 
        target_to_memory:  method that converts the task_data target into a location to record in memory
    
    """

    # TODO FIXME!  PoseNode instead of LocationNode
    # TODO put ref object info here?
    def __init__(self, agent, task_data):
        super().__init__(agent)
        assert task_data.get("target") is not None
        loc = self.target_to_memory(task_data["target"])
        loc_memid = LocationNode.create(agent.memory, loc)
        TripleNode.create(
            agent.memory, subj=self.memid, pred_text="task_reference_object", obj=loc_memid
        )

    def target_to_memory(self, target):
        raise NotImplementedError


def maybe_task_list_to_control_block(maybe_task_list, agent):
    # if input is a list of tasks with len > 1, outputs a ControlBlock wrapping them
    # if it is a list of tasks with len = 1, returns that task
    if len(maybe_task_list) == 1:
        return maybe_task_list[0]
    if type(maybe_task_list) is not list:
        return maybe_task_list
    return ControlBlock(agent, {"new_tasks": maybe_task_list})
