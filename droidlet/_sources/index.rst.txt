


*********
droidlet
*********

droidlet is a modular embodied agent architecture, and a platform for building modular agents.  It is designed for integrating machine-learned and heuristically scripted components.  It includes

* APIs linking memory, action, perception, and language
* Perceptual and natural-language-understanding models
* Fully instantiated agents for the `locobot <http://www.locobot.org/>`__ and Minecraft
* Suite of data annotation tools.


Warning!  This is a research project, and the architecture itself is a research artifact.




Overview of the Droidlet Agent
------------------------------

The abstract droidlet agent consists of four major components: a :ref:`memory_label` system, a :ref:`controller_label`, a set of :ref:`perception_label` modules, and a set of low level :ref:`tasks_label`.  In a nutshell,


* the memory system acts as a nexus of information for all the other systems,
* the controller places tasks on a stack, based on the state of the memory system,
* the perceptual modules process information from the outside world and store it in the memory system,
* and the low-level tasks effect changes in the outside world.

The components in this library can be used separately from the "complete" agent; or can be replaced or combined as a user sees fit.
The high level control loop of the agent is cartooned in `here <https://github.com/facebookresearch/fairo/blob/main/agents/core.py>`__ and shown below:

.. code-block::

   while forever:
        run perception and update memory
        maybe place tasks on the stack, based on memory
        step topmost task on Task Stack

Controller
----------

Instead of directly affecting the agent's environment, the controller places `Task <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/task.py>`__ objects on a `Task Container <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/task.py#L219>`__.   In order to choose which to place (if any), it needs to read information about the state of the world from memory.

In addition to the "abstract" droidlet agent, this repo has two "batteries included" droidlet agent instantiations, located `here <https://github.com/facebookresearch/fairo/tree/main/agents/locobot>`__ and `here <https://github.com/facebookresearch/fairo/tree/main/agents/craftassist>`__.  In these, the controller is mediated in part by a Dialogue Manager with an associated Dialogue Stack, which attempt to convert human interactions (dialogue) into specifications of Tasks.  The Dialogue Stack is populated with Dialogue Objects, which carry out chunks of human interaction, for example asking for a clarification about a command.

The Dialogue Manager is in turn powered by a neural semantic parser, which translates natural language into partially specified programs in a DSL decribed `here <https://github.com/facebookresearch/fairo/tree/main/droidlet/documents/logical_form_specification>`__.  The partially specified programs are then made "executable" (i.e. interpreted into Task Stack manipulations, including adding Tasks) by the Intepreter Dialogue Object.  This object is further broken down into subinterpreters that correspond to subtrees of the logical forms in the DSL.

The Controller operation in the droidlet agents can be sketched as:

.. code-block::

   if new command:
        logical_form = semantic_parser.translate(new command)
        interpret(logical_form, agent_memory)
   if TaskStack is empty:
        maybe place default behaviors on the stack

Here "default behaviors" might be running SLAM or other self-supervised exploration.

Memory
------

The `memory system </../../blob/master/docs/source/memory.md>`__ serves as the interface for passing information between the various components of the agent.  It consists of

**A database**\ , currently implemented in SQL, with an overlayed triplestore.  The entry point to the underlying SQL database is an `AgentMemory <https://github.com/facebookresearch/fairo/blob/main/droidlet/memory/sql_memory.py>`__ object.  The database can be directly queried through SQL; some common queries using triples or that otherwise are messy in raw SQL have been simplified and packaged.

**MemoryNodes**\ , which are Python wrappers for coherent data.  MemoryNodes collate data about a particular entity or event.  There are MemoryNodes for ReferenceObjects (things that have a location in space), for Tasks, for chats and commands, etc.

**FILTERS** objects, which connect the AgentMemory and MemoryNodes to the DSL used in the droidlet agents

Tasks
-----

The `Task <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/task.py>`__ objects abstract away the difficulties of actually carrying out the tasks, and allow a uniform interface to the controller and memory across different platforms.

Task objects define a .step() method; on each iteration through the main agent loop, the Task is stepped, and the Task Stack steps its highest priority Task.
The Task objects themselves are *not* generic across agents; and can be heuristic or learned.  In the current droidlet agent controller, they are registered `here <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/interpreter.py#L83>`__\ , for example `here <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/craftassist/mc_interpreter.py#L88>`__ and `here <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/robot/loco_interpreter.py#L71>`__

The `Task Container <https://github.com/facebookresearch/fairo/blob/main/droidlet/interpreter/task.py#L219>`__ is maintained by the Memory system, and provides methods for examining and manipulating Task Objects

Perception
----------

Perceptual modules process information about the agent's environment and write to memory.  Each perceptual module should have a .perceive() method, which is called
`here <https://github.com/facebookresearch/fairo/blob/main/agents/loco_mc_agent.py#L281>`__\ , during the main agent loop.



More in Depth:

--------------

.. toctree::
   :maxdepth: 2

   controller
   memory
   perception
   tasks
   droidlet_agents
