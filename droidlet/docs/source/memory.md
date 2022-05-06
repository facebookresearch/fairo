```eval_rst
.. _memory_label:
```
# Memory
The memory system serves as the interface for passing information between the various components of the agent.  It consists of a database, currently implemented in SQL, with an overlayed triplestore, and MemoryNodes, which are Python wrappers for coherent data.  FILTERS objects connect the AgentMemory and MemoryNodes to the controller.

## Database #

The entry point to the underlying SQL database is an [AgentMemory](https://github.com/facebookresearch/fairo/blob/main/droidlet/memory/sql_memory.py) object.
The database can be directly accessed by [\_db\_read\(query, \*args\)](https://github.com/facebookresearch/fairo/blob/main/droidlet/memory/sql_memory.py#L958).  Some common queries using triples or that are otherwise unwieldy in raw SQL have been packaged in the [basic\_search\(search_data\)](https://github.com/facebookresearch/fairo/blob/main/droidlet/memory/sql_memory.py#L354) [interface](https://github.com/facebookresearch/fairo/blob/main/droidlet/memory/memory_filters.py#L198).

```eval_rst
 .. autoclass:: droidlet.memory.sql_memory.AgentMemory
  :members:     basic_search, get_mem_by_id, forget, _db_read, _db_write, get_time, get_world_time, get_recent_entities, add_triple, tag, untag, get_memids_by_tag, get_tags_by_memid, get_triples, task_stack_push, task_stack_update_task, task_stack_peek, task_stack_pop, task_stack_pause, task_stack_clear, task_stack_resume, task_stack_find_lowest_instance, get_last_finished_root_task
```

## MemoryNodes #
MemoryNodes are python objects that collate data about a particular entity or event.  There are MemoryNodes for ReferenceObjects (things that have a location in space), for Tasks, for chats and commands, etc.  MemoryNode .__init__ calls take a memid (key in the base Memories table in the database).  To create a memid (and to enter information relevant to the MemoryNode)  use the classes' `.create()` method.  These have a different input signature for each type of MemoryNode, but always output a memid.


```eval_rst
.. autoclass:: droidlet.memory.memory_nodes.MemoryNode
.. autoclass:: droidlet.memory.memory_nodes.ProgramNode
.. autoclass:: droidlet.memory.memory_nodes.NamedAbstractionNode
.. autoclass:: droidlet.memory.memory_nodes.ReferenceObjectNode
.. autoclass:: droidlet.memory.memory_nodes.PlayerNode
.. autoclass:: droidlet.memory.memory_nodes.SelfNode
.. autoclass:: droidlet.memory.memory_nodes.LocationNode
.. autoclass:: droidlet.memory.memory_nodes.AttentionNode
.. autoclass:: droidlet.memory.memory_nodes.TimeNode
.. autoclass:: droidlet.memory.memory_nodes.ChatNode
.. autoclass:: droidlet.memory.memory_nodes.TaskNode
```
