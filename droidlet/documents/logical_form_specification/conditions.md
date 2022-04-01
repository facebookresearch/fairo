Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# Conditions #

expand "terminate_condition" to "conditions" more generally (e.g. allow "if condition" statements outside of loops).  Add an optional "init_condition" internal node to all actions.  This allows things like "move to the house and jump every third step". 
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "actions" : ACTION_LIST			    
}
```
Where 

```
CONTROL = {
      "init_condition": CONDITION, 
      "terminate_condition": CONDITION
      }
```

```
SPATIAL_CONTROL = { 
    "direction_information": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 
                             'AROUND' / 'CLOCKWISE' / 'ANTICLOCKWISE' 
      
}
```

Note: `terminate_condition` now covers number of times an action will be repeated as well.
`init_condition` implies: when CONDITION (the value of `init_condition`) happens do action.
`terminate_condition` implies: keep doing action until CONDITION (the value of `terminate_condition`) is met.

control inside event_sequence implies control (with either terminate_condition, on a certain condition and along a certain direction) on the sequence of events.
For a single action, the event_sequence will be a list of length 1. 

If the semantic parser outputs only a raw CONTROL dict, the interpreter should apply that to the action at the top of the bot's stack.
CONDITION will be more explicitly defined by what works with blockly, but will be a superset of current stop conditions


```
ACTION_LIST = { "list": [ACTION, ..., ACTION],
	        "control": CONTROL,
          	"spatial_control" : SPATIAL_CONTROL
	       }
```

```
ACTION = {"action_type" : "DIG"/"BUILD"/ ...,
	  ...,
	  "action_list" : ACTION_LIST,
	  "control": CONTROL,
	  "spatial_control" : SPATIAL_CONTROL
	  }
	  
```
The "action_type" key specifies a single action and is mutually exclusive with the "action_list" key, which allows nesting.


## CONDITION ##
```
CONDITION = {
  "condition_type" : 'MEMORY' / 'COMPARATOR' / 'NEVER' / 'TIME' / 'ACTION' /
                     'AND' / 'OR' / 'NOT'
  "condition": MEMORY_CONDITION/COMPARATOR/TIME_CONDITION/ACTION_CONDITION/AND_CONDITION/OR_CONDITION/NOT_CONDITION
  "condition_span": span}
```
AND OR and NOT modify other conditions:
```
 AND_CONDITION = {"and_condition": [CONDITION, ..., CONDITION]} 
```
```
 OR_CONDITION = {"or_condition": [CONDITION, ..., CONDITION]} 
```
```
 NOT_CONDITION = {"not_condition": CONDITION} 
```
```
ACTION_CONDITION = {has_type: "BUILD"/"MOVE"/...,
		    associated_chat: "THIS",
		    "time":  "CURRENT"
}
```

ACTION_CONDITION is true when there is a task on the stack matching the conditions.  logical_form "THIS" means the task is associated to the same chat as the command defining the ACTION_CONDITION.   "time" "CURRENT" is true while the (toplevel) task on the stack when the ACTION_CONDITION remains.   Eventually this will be expanded to FILTERS.
```
MEMORY_CONDITION = {
  'memory_exists': FILTERS, 
  'memory_modified': FILTERS}
```
```

TIME_CONDITION = {
    "comparator": COMPARATOR,
    "special_time_event" : 'SUNSET / SUNRISE / DAY / NIGHT / RAIN / SUNNY',
    "event": CONDITION,
  }
```
Note that `event` in time_condition represents when to start the timer and
`input_left` by default in time condition marks the time since the event condition.


## TerminateCondition ##
```
"terminate_condition": {"condition": {"comparison_type": "EQUAL",
                                    "input_left": {
                                      "filters": {"output": {"attribute": "RUN_COUNT"},
                                                  "special": {"fixed_value": "THIS"}}},
                                    "input_right": {"value": span}},
                      "condition_type": "COMPARATOR"}
```		      
