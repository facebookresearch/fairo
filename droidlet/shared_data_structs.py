from .interpreter.condition import AlwaysCondition, NeverCondition, NotCondition, TaskStatusCondition
from .memory.memory_nodes import TaskNode, TripleNode


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

    def add_child_task(self, t, prio=1):
        TaskNode(self.agent.memory, self.memid).add_child_task(t, prio=prio)

    def __repr__(self):
        return str(type(self))


def get_default_conditions(task_data, agent, task):
    """
    takes a task_data dict and fills in missing conditions with defaults

    Args:
        task_data (dict):  this function willtry to use the values of "init_condition",
                           "stop_condition", "run_condition", and "remove_condition"
        agent (Droidlet Agent): the agent that is going to be doing the Task controlled by
                                condition
        task (droidlet.shared_data_structs.Task):  the task to be controlled by the conditions
    """
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