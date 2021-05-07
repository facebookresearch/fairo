This document is a high level walkthrough of memory.

## High level pipeline

We have 5 important components to think about when we think of a droidlet agent:

1. **Language understanding component** that takes in language and converts that to a partially executable program (we call these programs logical form).  Example:

```
Input: "go to the house"

Output: 
{
"dialogue_type": "HUMAN_GIVE_COMMAND", 
"action_sequence": [
    {"location": {
        "reference_object": {
             "filters": {
                    "triples": [{"pred_text": "has_name", "obj_text": "house" }]}}},
     "action_type": "MOVE"
    }
   ]
}
```

2. **Interpreter** - this component takes in output of step #1 and tries to convert that into an executable program (we call this a `Task`) by interfacing with memory to fill in world specific information. For example:

```
Input:
{
"dialogue_type": "HUMAN_GIVE_COMMAND", 
"action_sequence": [
    {"location": {
        "reference_object": {
             "filters": {
                    "triples": [{"pred_text": "has_name", "obj_text": "house" }]}}},
     "action_type": "MOVE"
    }
   ]
}

Output:
task_data = {"target": <Absolute coordinates of house> , "action_dict": <Input from above>}
Move(agent, task_data)
```


To get from the input to the output, the interpreter needs to fill in missing details in the logical form by stepping through it. The way we do this is, the interpreter steps throuhg the logical form step by step and asks for world-specific information from the memory (through agent) if needed. Here, it first checks the `dialogue_type` (`HUMAN_GIVE_COMMAND`), then based on the `dialogue_type` it decides on how to read its siblings. In the example here, it first checks the the type of the `action` (`MOVE`) and based on action type it reads the children of the actions (`"location"` for a `MOVE` in this example).

While reading the nested dictionary, one key at a time, the interpreter decides when it needs to ask memory to populate some information from the world (in this case the exact coordinates of an object called `"house"`). Assuming that the perception component is already populating the agent's memory with what it knows, we should have a way of extracting the absolute coordinates of a `"house"`.

3. Perception : We won't go into the details of this component but the high level idea here is to get information from the world and store it in memory. This component interfaces with the agent’s world.

4. Memory: This component is the nexus of information for all other systems - any information that needs to be stored for later use, any information that needs to be communicated between modules, any information about the state of the world. 

5. Task: These are things that cause and affect changes in the world. Tasks are small self contained actions that an agent can perform, example : jumping, moving from x to y, pointing etc. Tasks are platform specific and under the hood implementation for a task can be different in different worlds and environments. For example: moving in robot is fundamentally different than moving in Minecraft. But the interface is same across.

## Memory

This document now dives into the current state of memory, the behind the scenes and the bridge between interpreter, memory and perception.

To start with, we have a concept of memory which is the nexus of all information that is important for the system of modules explained above to work together.

In theory, memory is a key-value store, where keys are unique identifiers and values are heterogenous memory objects or nodes. Retrieving a record or a property of a record might sometimes require deep inspection, multiples levels of querying, and might require significant non-cachable computation at retrieval time, for example: `"what is tallest thing I've made in the last 5 minutes ?"`. 

Here, we first need to query memory for all the things `I` constructed in the last 5 minutes, then I need to extract the height property of each of those things and then rank them in decreasing order of height and pick the first one.

Note that some parts of the computation above are done outside the scope of just querying a key-value store, for example ranking on height and picking the first one - these are handled by the interpreter after getting the output of queries over memory.

When we think of memory, there are few salient components:
1. The implementation - SQL schema - the actual tables that hold the information.
2. Interfacing with the SQL database - this is the piece that write raw SQL queries to read from and write to the SQL database.
3. Memory nodes - Instance of a memory object
4. High level memory APIs - 

Pseudo code of how it works is:

On a high level each of the things mentioned above can be expanded by a new agent



### Implementation
Right now we have implemented the key-value store using sqlite. We have a shared base schema here which can be extended by a new agent depending on the use case. To extend:
SCHEMAS defined in : base_memory_schema.sql + mc_memory_schema.sql

### Memory Nodes
Each Memory node is an instance of a memory and can also be extended on top of shared memory nodes available. To extedn :
Memory_nodes:
Every node has TABLE_COLUMNS, TABLE, and NODE_TYPE. TABLE and TABLE_COLUMNS come from schema 
NODE_TYPE ?
And some class attributes like: name, triple
Create, update, select, getters
Some class attributes are exposed
Memory itself has memory nodes. Some are shared: base_agent.memory_nodes
Some agent specific: mc_memory_nodes

Memory node:
To create memory node:
Memory, attributes for this node.
To do anything else - go through agent, so self.agent_memory._db_read_one()

All the tables for the memory nodes are in the schema + base_schema



### Agent memory
This is the high level memory interface that creates the database, given the schema. This is the class that supports all raw SQL database access for read, writes and updates to the database 
_db_read
_db_read_one
db_write
_db_write

query memory, update, remove, upsert, load things into memory/sql table, set, update, add
Memory_filters.py -> structured searches on tables in memory
Basic_search
self.memory = MCAgentMemory(load_minecraft_specs=False, agent_time=T)
Init:
  self,
       db_file=":memory:",
       db_log_path=None,
       schema_paths=SCHEMAS,
       load_minecraft_specs=True,
       load_block_types=True,
       load_mob_types=True,
       preception_range=PERCEPTION_RANGE,
       agent_time=None,


To extend:
To tag: 
memory.tag(memid, "VOXEL_OBJECT")
      
To get tags:
memory.get_triples
(subj=memid, pred_text=k, obj_text=v)

To get from db:
memory._db_read("SELECT x, y, z FROM VoxelObjects WHERE uuid=?", self.memid)

To read one:
x, y, z = self.agent_memory._db_read_one(
           "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
       )
To create uuid:
memid = cls.new(memory)

To insert :
memory.db_write(
           "INSERT INTO ReferenceObjects(uuid, eid, x, y, z, yaw, pitch, ref_type, type_name, player_placed, agent_placed, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
           memid,
           mob.entityId,
           mob.pos.x,
           mob.pos.y,
           mob.pos.z,
           mob.look.yaw,
           mob.look.pitch,
           "mob",
           mobtype,
           player_placed,
           agent_placed,
           memory.get_time(),
       )

To add_triple:
memory.add_triple(subj=memid, pred_text="has_name", obj_text=mobtype)
 ? this vs memory.tag ?


### Interface


### Usage

[AgentMemory]() provides us the interface to access the memory.
Under the hood there is a sql database, the schema for which is [here]()
We have MemoryNodes

Finally, there is a very abstract object (0), which is the "memory model" used by the spec.  This has not been carefully formalized, but looks something like: the memory is a collection of heterogenous records (memories), with a finite index (we call them memids).  

Right now you can go through high level memory APIs or memory node.get()