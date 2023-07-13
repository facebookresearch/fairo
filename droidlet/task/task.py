"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.task.condition_classes import (
    AlwaysCondition,
    TaskRunCountCondition,
    SwitchCondition,
    AndCondition,
)
from droidlet.memory.memory_nodes import TaskNode, LocationNode, TripleNode

# FIXME TODO store conditions in memory (new table)
# TaskNode method to update a tasks conditions
# dsl/put_memory for commands to do so

# FIXME agent, move this to a better place
# from droidlet.shared_data_structs import Task


def maybe_bundle_task_list(agent, task_list):
    """
    utility to convert a list of task into a new_tasks generator.
    """
    task_gens = []
    for t in task_list:
        task_gens.append(task_to_generator(t))

    if len(task_gens) > 1:
        return task_to_generator(ControlBlock(agent, {"new_tasks": task_gens}))
    elif len(task_gens) == 1:
        return task_gens[0]
    else:
        return None


def maybe_update_condition_memid(condition, memid, pos="value_left"):
    """
    update the condition to use the memid of the Task it is controlling;
    it was interpreted with a "THIS" keyword
    """
    if hasattr(condition, pos):
        v = getattr(condition, pos)
        if hasattr(v, "memory_filter"):
            if hasattr(v.memory_filter.head, "memid"):
                if v.memory_filter.head.memid == "NULL":
                    # this was a special "THIS" filter condition, needed to wait till here
                    # to get memid
                    v.memory_filter.head.memid = memid


class task_to_generator:
    def __init__(self, task):
        self.task = task
        self.fuse = SwitchCondition(task.agent.memory)
        self.fuse.set_status(False)
        self.task.init_condition = AndCondition(
            task.agent.memory, [task.init_condition, self.fuse]
        )
        TaskNode(task.agent.memory, task.memid).update_task(task=task)

    def __call__(self):
        self.task.reset()
        return self.task


class Task(object):
    """This class represents a Task, the exact implementation of which
    will depend on the framework and environment. A task can be placed on a
    task stack, and represents a unit (which in itself can contain a sequence of s
    smaller subtasks).

    Attributes:
        memid (string): Memory id of the task in agent's memory
        interrupted (bool): A flag indicating whether the task has been interrupted
        finished (bool): A flag indicating whether the task finished
        name (string): Name of the task
        undone (bool): A flag indicating whether the task was undone / reverted
        last_stepped_time (int): Timestamp of last step through the task

    Examples::
        >>> Task()
    """

    def __init__(self, agent, task_data={}, memid=None):
        self.agent = agent
        self.run_count = 0
        self.reset()
        self.undone = False
        if memid:
            self.memid = memid
            N = TaskNode(agent.memory, self.memid).update_task(task=self)
            # this is an egg, hatch it
            if N.prio == TaskNode.EGG_PRIO:
                N.get_update_status({"prio": TaskNode.CHECK_PRIO})
        else:
            TaskNode.create(self.agent.memory, self)

        # FIXME remove this entirely, always wrap toplevel Events in ControlBlock,
        #   only ControlBlocks need conditions
        self.init_condition = AlwaysCondition(None)

        # remember to get children of blocking tasks (and delete this comment)
        if task_data.get("blocking"):
            TripleNode.create(
                self.agent.memory, subj=self.memid, pred_text="has_tag", obj_text="blocking_task"
            )

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
            if self.finished:
                TaskNode(self.agent.memory, self.memid).get_update_status(
                    {"prio": TaskNode.FINISHED_PRIO, "finished": True}
                )
                return
            query = "SELECT MEMORY FROM Task WHERE ((prio>=1) AND (_has_parent_task=#={}))".format(
                self.memid
            )
            _, child_task_mems = self.agent.memory.basic_search(query)
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

    def reset(self):
        """
        reset the Task.
        this may require re-interpretation, e.g. if a reference object needs
        to be recomputed.  Set this up in override
        """
        self.interrupted = False
        self.finished = False
        self.name = None
        self.last_stepped_time = None
        self.prio = TaskNode.CHECK_PRIO

    def add_child_task(self, t, prio=TaskNode.CHECK_PRIO + 1):
        TaskNode(self.agent.memory, self.memid).add_child_task(t, prio=prio)

    def __repr__(self):
        return str(type(self))


# if you want a list of tasks, have to enclose in a ControlBlock
# if you want to loop over a list of tasks, you need a tasks_fn that
# generates a ControlBlock C = tasks_fn() wrapping the (newly_generated) list,
# and another D that encloses C, that checkes the terminate condition
#
# FIXME/TODO: name any nonpicklable attributes in the object
class ControlBlock(Task):
    """Container for task control

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
            task_data["new_tasks"] is a list of callables, when called they returns a Task or None
            when it returns None, this ControlBlock is finished.
            to make an infinite loop, the callable needs to keep returning Tasks;
    """

    def __init__(self, agent, task_data):
        super().__init__(agent, task_data=task_data)

        # TODO put these in memory in a new table?
        # TODO methods for safely changing these
        i, t = self.get_default_conditions(task_data, agent)
        self.init_condition = i
        self.terminate_condition = t

        task_fns = task_data.get("new_tasks")
        # TODO remove this, always pass in a list of task_generators
        # TODO handle extra info for resets
        # (including and up to re-interpreting lf, and DSL support for
        # forcing re-interpretation)
        if type(task_fns) is not list:
            task_fns = [task_fns]
        self.task_fns = task_fns
        TaskNode(self.agent.memory, self.memid).update_task(task=self)

    def get_default_conditions(self, task_data, agent):
        """
        takes a task_data dict and fills in missing conditions with defaults

        Args:
            task_data (dict):  this function will try to use the values of "init_condition" and  "terminate_condition"
            agent (Droidlet Agent): the agent that is going to be doing the Task controlled by
                                    condition
            task (droidlet.tasks.Task):  the task to be controlled by the conditions
        """
        init_condition = task_data.get("init_condition") or AlwaysCondition(None)
        terminate_condition = task_data.get("terminate_condition") or TaskRunCountCondition(
            agent.memory, self.memid, N=1
        )
        # check/maybe update if special "THIS" filter condition
        maybe_update_condition_memid(terminate_condition, self.memid)
        maybe_update_condition_memid(init_condition, self.memid)

        return init_condition, terminate_condition

    # WARNING: no TaskNode.step_wrapper... doing this by hand to propagate terminate condition
    def step(self):
        self_mem = TaskNode(self.agent.memory, self.memid)
        # set prio higher than children, we want this run first
        # todo if nested controls, keep track of prios and organize
        # FIXME update the setter so this happens automatically
        self_mem.get_update_status({"prio": TaskNode.CHECK_PRIO + 2})
        if self.terminate_condition.check():
            self.finished = True
            # propagate to children.  TODO some machinery/DSL to be more delicate
            for task_mem in self_mem.all_descendent_tasks(include_root=True):
                task_mem.get_update_status({"finished": True})
            return
        query = "SELECT MEMORY FROM Task WHERE ((prio>=1) AND (_has_parent_task=#={}))".format(
            self.memid
        )
        _, child_task_mems = self.agent.memory.basic_search(query)
        if (
            child_task_mems
        ):  # this task has active children, don't step self, let agent step children
            return

        if not self.finished:
            # check if we have reached the end of our task list, go back to beginning
            if self.task_list_idx == len(self.task_fns):
                self.task_list_idx = 0
                self.run_count += 1
                self_mem.update_task(task=self)
            # can only be here if
            # there is no child, so previous generated child task is finished.
            # start the next child in the sequence:
            g = self.task_fns[self.task_list_idx]
            # if g is a Task that has been wrapped via task_to_generator, light its fuse
            # if not, we assume that it generates a new Task that has not been activated in agent.task_step()
            # FIXME! force this assumption
            if hasattr(g, "fuse"):
                g.fuse.set_status(True)
            t = g()
            if t is not None:
                self.add_child_task(t, prio=None)
            self.task_list_idx = self.task_list_idx + 1

        self_mem.update_task(task=self)

    def reset(self):
        super().reset()
        self.task_list_idx = 0


class BaseMovementTask(Task):
    """a Task that changes the location of the agent

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
