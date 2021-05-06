This document is a high level walkthrough of memory.

We have 5 important components to think about when we think of the agent:
1. Language understanding component that takes in language and converts that to a partially executable program (we call this logical form).  Example:
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

2. Interpreter - this component takes in output of #1 and tries to convert that into an executable program (called a `Task`) by interfacing with memory to fill in world specific information. For example:

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
task_data = {"target": pos_of_house, "action_dict": input_above}
Move(agent, task_data)
```

Default to where Speaker is looking if no location is found. More on this in a doc about interpreter.
To get from the input to output, we need to fill in missing details in the logical form. In this case, the interpreter first reads the dialogue_type (HUMAN_GIVE_COMMAND), then based on dialogue_type, it reads its siblings - the type of action (MOVE) and based on action type - the children of the actions ("location" in this case: https://github.com/facebookresearch/droidlet/blob/main/base_agent/dialogue_objects/interpreter.py#L163 )
While reading the nested dictionary, interpreter decides when it needs to ask memory to populate information from the world (in this case exact coordinates of the "house"). Assuming that perception is already populating memory with what it knows, we should have a way of extracting the coordinates of a thing called a "house".

3. Perception : We won't go into the details of this component but the high level idea here is to get information from the world and store it in memory. This component interfaces with the agent’s world.

4. Memory: This component is the nexus of information for all other systems - any information that needs to be stored for later use, any information that needs to be communicated between modules, any information about the state of the world. 

5. Task: These are things that cause changes in the world.  Small self contained actions that agent performs : jumping, moving from x to y, pointing - platform specific, under the hood implementation different in different world. But the interface is same across. Moving in robot is fundamentally different than moving in Minecraft.

This document now dives into the current state of memory, the behind the scenes and the bridge between interpreter and memory.

To start with, we have a concept of memory which, in theory, is a key-value store, where keys are unique identifiers and values are memory objects.
[AgentMemory]() provides us the interface to access the memory.
Under the hood there is a sql database, the schema for which is [here]()
We have MemoryNodes

Finally, there is a very abstract object (0), which is the "memory model" used by the spec.  This has not been carefully formalized, but looks something like: the memory is a collection of heterogenous records (memories), with a finite index (we call them memids).  Retrieving a record or a property of a record might require deep inspection, and might require significant non-cachable computation at retrieval time, for example "what is inside the circle I just made?".   Indeed, it might require computation over a large set of records "what is the biggest thing to my left?"
