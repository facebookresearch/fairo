Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# Conditions #

expand "stop_conditions" to "conditions" more generally (e.g. allow "if condition" statements outside of loops).  Add an optional "on_condition" internal node to all actions.  This allows things like "move to the house and jump every third step". 
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "actions" : ACTION_LIST			    
}
```
Where 

```
CONTROL = {
      "repeat_count" : span,
      "repeat_key" : 'FOR'/'ALL',
      "on_condition": CONDITION, 
      "stop_condition": CONDITION
      }
```

```
SPATIAL_CONTROL = { 
    "direction_information": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 
                             'AROUND' / 'CLOCKWISE' / 'ANTICLOCKWISE' 
      
}
```

Note: `stop_condition` now covers number of times an action will be repeated as well.
`on_condition` implies: when CONDITION (the value of `on_condition`) happens do action.
`stop_condition` implies: keep doing action until CONDITION (the value of `stop_condition`) is met.

control inside action implies control (with either stop_condition, on a certain condition and along a certain direction) on each instance of the action.
control outside the action list implies control over the entire sequence of action.

If TTAD outputs only a raw CONTROL dict, the interpreter should apply that to the action at the top of the bot's stack.
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
