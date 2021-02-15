"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from condition import NeverCondition, AlwaysCondition, TaskStatusCondition
from memory_nodes import TaskNode

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

    def __init__(self, agent):
        self.agent = agent
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
        self.stop_condition = NeverCondition(None)
        self.on_condition = AlwaysCondition(None)
        self.remove_condition = TaskStatusCondition(agent, self.memid)
        self.child_generator = TaskGenerator()
        TaskNode(agent.memory, self.memid).update_task(task=self)

    @staticmethod
    def step_wrapper(stepfn):
        def modified_step(self):
            if self.remove_condition.check():
                self.finished = True
            if self.finished:
                TaskNode(self.agent.memory, self.memid).get_update_status(
                    {"prio": -1, "finished": True}
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


# FIXME new_tasks_fn --> new_tasks
# FIXME/TODO: name any nonpicklable attributes in the object
# should we just have this be the behavior of the step of an unsublcassed Task object?
class ControlBlock(Task):
    """Container for task control
    

    Args:
        agent: the agent who will perform this task
        task_data (dict): a dictionary stores all task related data
    """

    def __init__(self, agent, task_data):
        super().__init__(agent)
        for task in task_data.get("new_tasks_fn"):
            self.child_generator.append(task)
        self.stop_condition = task_data.get("stop_condition", NeverCondition(None))
        self.on_condition = task_data.get("on_condition", AlwaysCondition(None))
        self.remove_condition = task_data.get(
            "remove_condition", TaskStatusCondition(agent, self.memid)
        )
        TaskNode(self.agent.memory, self.memid).update_task(task=self)

    @Task.step_wrapper
    def step(self):
        try:
            t = next(self.child_generator)
        except StopIteration:
            t = None
        if t:
            self.add_child_task(t)
        else:
            self.finished = True


# the extra complexity here is that we are
# allowing loops that are finite sequences, and loops
# driven by a task generator, AND allowing child tasks to be
# appended to a parent after the parent is placed in memory or is active
# Probably can clean this up
# without building this mini-stack,
# just by using stop/remove conditions.... TODO?
class TaskGenerator:
    """
    a gadget for storing child tasks.  the child tasks are stored in a 
    list; and each entry in the list can be a Task, or a arg-less callable that outputs Task objects

    Args:

    Attributes:
        tasks (list): child (generator) list
        count (int): index into child (generator) list     
        __next__: get the Task from tasks[self.count].  if tasks[self.count] is a Task, 
                  return it; else if it is a callable, call it (and expect it to return 
                  a Task object, which __next__ in turn returns
    """

    def __init__(self):
        self.count = 0

        # this one for debug purposes, counts actual tasks output
        # not actually used for anything rn
        self.tasks_added = 0

        self.tasks = []

    def append(self, tasks):
        if type(tasks) is list:
            self.tasks.extend(tasks)
        else:
            self.tasks.append(tasks)

    def __len__(self):
        return len(self.tasks) - self.count

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.tasks):
            raise StopIteration
        task_gen = self.tasks[self.count]
        task = None
        if callable(task_gen):
            # WARNING: getting a task from the generator does not increment count!
            task = task_gen()
            if task:
                self.tasks_added += 1
                return task
            else:
                self.count += 1
                return next(self)
        else:
            task = self.tasks[self.count]
            self.count += 1
            self.tasks_added += 1
            return task
