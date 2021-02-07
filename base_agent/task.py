"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from condition import NeverCondition

DEFAULT_THROTTLING_TICK = 16
THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

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

    def __init__(self):
        self.memid = None
        self.interrupted = False
        self.finished = False
        self.name = None
        self.undone = False
        self.last_stepped_time = None
        self.throttling_tick = DEFAULT_THROTTLING_TICK
        self.stop_condition = NeverCondition(None)

    def step(self, agent):
        """The actual execution of a single step of the task is defined here."""
        # todo? make it so something stopped by condition can be resumed?
        if self.stop_condition.check():
            self.finished = True
            return
        return

    def add_child_task(self, t, agent, pass_stop_condition=True):
        """Add a child task to the task_stack and pass along the id
        of the parent task (current task)"""
        # FIXME, this is ugly and dangerous; some conditions might keep state etc?
        if pass_stop_condition:
            t.stop_condition = self.stop_condition
        agent.memory.task_stack_push(t, parent_memid=self.memid)

    def interrupt(self):
        """Interrupt the task and set the flag"""
        self.interrupted = True

    def check_finished(self):
        """Check if the task has mark itself finished

        Returns:
            bool: If the task has finished
        """
        if self.finished:
            return self.finished

    def hurry_up(self):
        """Speed up the task execution"""
        self.throttling_tick /= 4
        if self.throttling_tick < THROTTLING_TICK_LOWER_LIMIT:
            self.throttling_tick = THROTTLING_TICK_LOWER_LIMIT

    def slow_down(self):
        """Slow down task execution"""
        self.throttling_tick *= 4
        if self.throttling_tick > THROTTLING_TICK_UPPER_LIMIT:
            self.throttling_tick = THROTTLING_TICK_UPPER_LIMIT

    def __repr__(self):
        return str(type(self))
