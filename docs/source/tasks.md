```eval_rst
.. _tasks_label:
```

# Tasks

Task objects abstract away the difficulties of actually carrying out the tasks, and allow a uniform interface to the controller and memory across different platforms.

Task objects define a .step() method; on each iteration through the main agent loop, the Task is stepped, and the Task Stack steps its highest priority Task.
The Task objects themselves are *not* generic across agents; and can be heuristic or learned.  In the current droidlet agent controller, they are registered [here](https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/interpreter.py#L83), [here](https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/craftassist/mc_interpreter.py#L88) and [here](https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/robot/loco_interpreter.py#L71)

The base Task object is

```eval_rst
 .. autoclass:: droidlet.interpreter.task.Task
  :members:    step, add_child_task, interrupt, check_finished
```


The Task Stack is maintained by the Memory system, and provides methods for examining and manipulating Task Objects. See the task_stack functions in [memory](memory.md).
