```eval_rst
.. _tasks_label:
```

# Tasks

Task objects abstract away the difficulties of actually carrying out the tasks, and allow a uniform interface to the controller and memory across different platforms.

Task objects define a .step() method; on each iteration through the main agent loop, the Task is stepped, and the Task Stack steps its highest priority Task.
The Task objects themselves are *not* generic across agents; and can be heuristic or learned.  In the current droidlet agent controller, they are registered [here](https://github.com/fairinternal/minecraft/blob/master/base_agent/dialogue_objects/interpreter.py#L76), [here](https://github.com/fairinternal/minecraft/blob/master/craftassist/agent/dialogue_objects/mc_interpreter.py#L78) and [here](https://github.com/fairinternal/minecraft/blob/master/locobot/agent/dialogue_objects/loco_interpreter.py#L64)

The base Task object is

```eval_rst
 .. autoclass:: base_agent.task.Task
  :members:    step, add_child_task, interrupt, check_finished
```


The Task Stack is maintained by the Memory system, and provides methods for examining and manipulating Task Objects. See the task_stack functions in [memory](memory.md).
