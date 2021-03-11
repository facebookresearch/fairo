Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# EVENT draft #
``` 
COMMAND = {"action_sequence" : [ACTION/COMMAND, ...., ACTION/COMMAND],
           "init_condition": CONDITION,
	   "run_condition": CONDITION,
           "stop_condition": CONDITION,
           "remove_condition": CONDITION,
           "spatial_control": {"direction_information": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 
                             'AROUND' / 'CLOCKWISE' / 'ANTICLOCKWISE'}
	   }
```

The "init_condition" specifies when the command should be
considered for running.  That is, until the init_condition is satisfied, the "run_condition" is not checked.
the "remove_condition" being satisfied removes the command from consideration permanently; if the remove_condition is
satisfied the run_condition will never be checked again.  Thus init_condition and remove_condition are the window in which the event
can occur

The run_condition is checked every agent step after the init_condition has been satisfied if the task is not running; if it is satisfied, 
the next step in the action_sequence is set to priority > 0 (asks to be run by the agent).  The stop_condition is checked every agent step where the
current step of the action_sequence is set to priority > 0; if it is satisfied, the priority is set to 0
(tells agent to not run but check conditions).  The stop_condition takes precedence over the run_condition.  So in short:
Once the init_condition has been satisfied, the action_sequence is stepped when the run_condition is satisfied,
unless the stop_condition is satisfied, until the remove_condition is satisfied.

In pseudocode:

```python
started = False
finished = False
while True: #agent loop:
    # this block is executed independently for each not finished command:
    if not started:
        if init_condition:
            started = True
    if started and not finished 
        if run_condition:
	    agent_set_action_sequence_priority_to_runme
	if stop_condition:
	    agent_set_action_sequence_priority_to_check
	    
    if remove_condition:
        finished = True
```

Each condition key and the "spatial_control" blocks are optional. 
The defaults (if a key is not set) are "init_condition": ALWAYS, "run_condition": ALWAYS, "stop_condition": NEVER.

The default for "remove_condition" is a bit more involved. Note that conditions are checked at each agent.step(),
and an "ALWAYS" would then immediately remove the command.  While we can check conditions on the task, the FILTERS
for the task sequence would be different for every command... so we introduce a special value for use inside
a CONDITION in a "control" block: "this_sequence", referring to the command sequence that is a sibling of the "control" key.
The FILTER corresponding to "this_sequence" is
```
THIS_ACTION_SEQUENCE = {"memory_type": "TASKS",
                        "triples": [{pred_text:"has_tag", "obj_text": "THIS_SEQUENCE"}]}

```
(we will need to update this when we have "obj_text": {"fixed_value": "THIS_SEQUENCE"}).  Furthermore, for each Task in memory, we store the
number of times it has run; and so have the attribute "RUN_COUNT".  The default (that is, what would be interpreted
if the key is not set) for remove_condition is 
```
THIS_SEQUENCE_FINISHED = {
                           "comparator": {
			     "comparison_type": "EQUAL",
                             "input_left": {
			       "value_extractor": {
			         "filter": {
			           "memory_type": "TASKS",
				   "triples": [{pred_text:"has_tag", "obj_text": "THIS_SEQUENCE"}],
				   "output": {"attribute": "RUN_COUNT"}
				 }
			       }
			     },
			     "input_right": {
			       "value_extractor": {
			         "fixed_value": 1
			       }
			     }
			    }
			   }
			  }
```

For "repeat 10 times" the remove condition is
```
{
  "comparator": {
    "comparison_type": "EQUAL",
    "input_left": {
      "value_extractor": {
	"filter": {
	  "memory_type": "TASKS",
	  "triples": [{pred_text:"has_tag", "obj_text": "THIS_SEQUENCE"}],
	  "output": {"attribute": "RUN_COUNT"}
	}
      }
    },
    "input_right": {
      "value_extractor": {
	"span": [0, [1,1]]
      }
    }
  }
}

```

We don't _really_ need 4 conditions; and could probably merge run_condition and stop_conditions, but separating allows each of
the conditions to be simpler (for example when the condition for running is different from the condition for stopping).


TODO modifiers (faster/slower, happily, etc...) 
