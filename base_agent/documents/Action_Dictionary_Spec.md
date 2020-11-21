Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# Dialogue Types #

## Human Give Command Dialogue type ##
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : list(<CommandDict>)
}
```
The CommandDict for each action type is described in the [Action subsection]().

## Noop Dialogue Type ##
```
{ "dialogue_type": "NOOP"}
```

## Get Memory Dialogue Type ##
```
{
  "dialogue_type": "GET_MEMORY",
  "filters": <FILTERS>,
  "replace": true
}
```
## Put Memory Dialogue Type ##
```
{
  "dialogue_type": "PUT_MEMORY",
  "filters": <FILTERS>,
  "upsert" : {
      "memory_data": {
        "memory_type": "REWARD" / "TRIPLE",
        "reward_value": "POSITIVE" / "NEGATIVE",
        "has_tag" : span,
        "has_colour": span,
        "has_size": span
      } }
}
```

## Actions ##

### Build Action ###
This is the action to Build a schematic at an optional location.

```
{ "action_type" : BUILD,
  <Location>,
  <Schematic>,
  <Repeat> (with repeat_key: 'FOR' and additional repeat_dir: 'SURROUND'),
  "replace" : True
}
    
```
### Copy Action ###
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

```
{ "action_type" : 'BUILD',
  <Location>,
  <ReferenceObject>,
  <Repeat> (repeat_key = 'FOR'),
  "replace" : True
}
```

### Spawn Action ###
This action indicates that the specified object should be spawned in the environment.
Spawn only has a name in the reference object.

```
{ "action_type" : 'SPAWN',
  "reference_object" : {
      "text_span" : span,
      <Repeat>(repeat_key= 'FOR'),
      "has_name" : span,
    },
    <Repeat>(repeat_key= 'FOR'),
    "replace": True
}
```

### Resume ###
This action indicates that the previous action should be resumed.

```
{ "action_type" : 'RESUME',
  "target_action_type": span
}
```

### Fill ###
This action states that a hole / negative shape needs to be filled up.

```
{ "action_type" : 'FILL',
  "has_block_type" : span,
  <ReferenceObject>,
  <Repeat>,
  "replace": True
}
```

#### Destroy ####
This action indicates the intent to destroy a block object.

```
{ "action_type" : 'DESTROY',
  <ReferenceObject>,
  <Repeat>,
  "replace": True
}
```

#### Move ####
This action states that the agent should move to the specified location.

```
{ "action_type" : 'MOVE',
  <Location>,
  <StopCondition>,
  <Repeat>,
  "replace": True
}
```

#### Undo ####
This action states the intent to revert the specified action, if any.

```
{ "action_type" : 'UNDO',
  "target_action_type" : span
}
```

#### Stop ####
This action indicates stop.

```
{ "action_type" : 'STOP',
  "target_action_type": span
}
```

#### Dig ####
This action represents the intent to dig a hole / negative shape of optional dimensions at an optional location.
The `Schematic` child in this only has a subset of properties.

```
{ "action_type" : 'DIG',
  <Location>,
  "schematic" : {
    "text_span" : span,
    <Repeat>(repeat_key = 'FOR'),
     "has_size" : span,
     "has_length" : span,
     "has_depth" : span,
     "has_width" : span
     },
  <StopCondition>,
  <Repeat>,
  "replace": True  
}
```

#### FreeBuild ####
This action represents that the agent should complete an already existing half-finished block object, using its mental model.

```
{ "action_type" : 'FREEBUILD',
  <ReferenceObject>,
  <Location>,
  <Repeat>,
  "replace": True
}
```

#### Dance ####
This action provides information to the agent to do a dance.
Also has support for Point / Turn / Look.

```
{ "action_type" : 'DANCE',
  <Location> (additional relative_direction values: ['CLOCKWISE', 'ANTICLOCKWISE']),
  <DanceType>
  "stop_condition" : {
      "condition_type" : NEVER,
  },
  "repeat" : {
    "repeat_key" : 'FOR',
    "repeat_count" : span, # Note no repeat_dir here.
  },
  "replace": True
}
```

#### Get ####
The GET action_type covers the intents: bring, get and give.

The Get intent represents getting or picking up something. This might involve first going to that thing and then picking it up in bot’s hand. The receiver here can either be the agent or the speaker (or another player).
The Give intent represents giving something, in Minecraft this would mean removing something from the inventory of the bot and adding it to the inventory of the speaker / other player.
The Bring intent represents bringing a reference_object to the speaker or to a specified location.

```
{
    "action_type" : 'GET',
    <ReferenceObject>,
    <Repeat>,
    "receiver" : <ReferenceObject> / <Location>
}
```

#### Scout ####
This command expresses the intent to look for / find or scout something.
```
{
    "action_type" : 'SCOUT',
    <ReferenceObject>
}
```

#### Modify action ####
Note that: This section is a temporary heristic based fix until we have a generative model that can handle "modify".

This command represents making a change or modifying a block object in a certain way.

[Examples](https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.o3lwuh3fj1lt)

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MODIFY',
      <ReferenceObject>,
      "modify_dict" :  THICKEN/SCALE/RIGIDMOTION/REPLACE/FILL
    }]
}
```
Where
```
THICKEN = {"modify_type": "THICKER"/"THINNER", "num_blocks": span}
SCALE = {"modify_type": "SCALE", "text_scale_factor": span, "categorical_scale_factor": "WIDER"/"NARROWER"/"TALLER"/"SHORTER"/"SKINNIER"/"FATTER"/"BIGGER"/"SMALLER"}
RIGIDMOTION = {"modify_type": "RIGIDMOTION", "categorical_angle": "LEFT"/"RIGHT"/"AROUND", "MIRROR"=True/False, "location": LOCATION}
REPLACE={"modify_type": "REPLACE", "old_block": BLOCK, "new_block": BLOCK, "replace_geometry": REPLACE_GEOMETRY}
FILL={"modify_type": "FILL"/"HOLLOW", "new_block": BLOCK}
```
And
```
BLOCK = {"has_x": span}
REPLACE_GEOMETRY = {"relative_direction": "LEFT"/"RIGHT"/"TOP"/"BOTTOM", "amount": "QUARTER"/"HALF"}
```
Note w2n doesn't handle fractions :(

If "location" is given in RIGIDMOTION, the modify is to move the reference object from its current location to the given location


### Subcomponents of action dict ###

#### Location ####
```
"location" : {
          "text_span" : span,
          "steps" : span,
          "has_measure" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          <ReferenceObject>,
          }
 ```

Note: for "relative_direction" == 'BETWEEN' the location dict will have two children: 'reference_object_1' : <ReferenceObject>['reference_object'] and 
'reference_object_2' : <ReferenceObject>['reference_object'] in the sub-dictionary to represent the two distinct reference objects.

#### Reference Object ####
```
"reference_object" : {
      "text_span" : span,
      <Repeat>,
      "special_reference" : 'SPEAKER' / 'AGENT' / 'SPEAKER_LOOK' / {'coordinates_span' : span},
      "filters" : <FILTERS>
      }
  } 
```
#### Stop Condition ####
```
"stop_condition" : {
      "condition_type" : 'ADJACENT_TO_BLOCK_TYPE' / 'NEVER',
      "block_type" : span
  }
```
#### Schematic ####

```
"schematic" : {
          "text_span" : span,
          <Repeat> (with repeat_key: 'FOR' and additional 'SURROUND' repeat_dir), 
          "has_block_type" : span,
          "has_name": span,
          "has_size" : span,
          "has_orientation" : span,
          "has_thickness" : span,
          "has_colour" : span,
          "has_height" : span,
          "has_length" : span,
          "has_radius" : span,
          "has_slope" : span,
          "has_width" : span,
          "has_base" : span,
          "has_depth" : span,
          "has_distance" : span,
      }
```
#### Repeat ####
```
"repeat" : {
            "repeat_key" : 'FOR'/ 'ALL'
            "repeat_count" : span,
            "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
      }
```

#### FACING ####
```
{
  "text_span" : span,
  "yaw_pitch": span,
  "yaw": span,
  "pitch": span,
  "relative_yaw" = {"angle": -360, -180, -135, -90, -45, 45, 90, 135, 180, 360 
  		    "yaw_span": span},
   "relative_pitch" = {"angle": -90, -45, 45, 90, 
  		    "pitch_span": span},
  "location": <LOCATION>
}
```

#### DanceType ####
```
"dance_type" : {
  "dance_type_name": span,
  "dance_type_tag": span,
  "point": <FACING>,
  "look_turn": <FACING>,
  "body_turn": <FACING>
}
```

#### Filters ####

TODO: Add "in_agent_view" like field to distinguish between "if you see a pig" vs "if there is a pig"
Short term bridge extending vqa+ref object specificity, and adding time filtering. To be torched when we get to seq2sql.  

```
FILTERS = { "output" : "memory" / "count" / ATTRIBUTE,
	    "contains_coreference": "yes",
      "memory_type": {"action_type": BUILD / DESTROY / DIG / FILL / SPAWN / MOVE} 
                     / "AGENT" / "REFERENCE_OBJECT" / "CHAT",
	    "argmax" : ARGMAX, 
	    "argmin" : ARGMIN,
	    "comparator": [COMPARATOR, ...],
	    "has_x": span / FILTERS, 
	    "author":  "AGENT" / "SPEAKER" / span,
	    "location": LOCATION }
```
In the `output` field we can either have a `memory` value which means return the entire memory node, or `count` that represents the equivalent of `count(*)`or a dictionary with `memory_node` representing the table name of memory node in sql and `attribute` representing the column name of that memory node (equivalent to `Select <attribute> from memory_node`).

`ATTRIBUTE` is :
```
{"attribute" : 'HEIGHT' / 'WIDTH' /  
'X' / 'Y' / 'Z' / 'REF_TYPE' / 'HEAD_PITCH' / 'HEAD_YAW' / 'BODY_YAW'/   
'NAME' / 'BORN_TIME' / 'MODIFY_TIME' / 'SPEAKER' / 'VISIT_TIME' /  
'FINISHED_TIME' / 'CHAT' / 'LOGICAL_FORM' /  'NAME' / 'SIZE' / 'COLOUR' / 
'ACTION_NAME' / 'ACTION_REFERENCE_OBJECT_NAME' / 'MOVE_TARGET' / 'LOCATION' / 
NUM_BLOCKS / LINEAR_EXTENT }
```
`COMPARATOR` is :
```
COMPARATOR = {
  "input_left" : {"value_extractor" : FILTERS / ATTRIBUTE / span }
  "comparison_type" : "GREATER_THAN" / "LESS_THAN" / "GREATER_THAN_EQUAL" / 
                       "LESS_THAN_EQUAL" / "NOT_EQUAL" / "EQUAL" / CLOSE_TO / MOD_EQUAL /
                       MOD_CLOSE_TO},
  "input_right" : {"value_extractor" : FILTERS / ATTRIBUTE, span }
  "comparison_measure" : span,
  "set_comparison": "ANY"/ "ALL"
}
CLOSE_TO = {"close_tolerance": "DEFAULT"/span}
MOD_EQUAL = {"modulus": "DEFAULT"/span}
MOD_CLOSE = {"modulus": "DEFAULT"/span, "close_tolerance": "DEFAULT"/span}
```
`MOD_EQUAL` in COMPARATOR is for e.g. handling time conditions like 'every 5 seconds' ,
`comparison_type` represents the kind of comparison (>, <, >=, != , =)
`input_left` is the candidate on left of the comparison,
`input_right` is the candidate on right of the comparison,
`input_left` or `input_right` has value ATTRIBUTE when the comparator is being used as part of a filter; and is searching over variables (and where the ATTRIBUTE of the variable is used for the filter).
`comparison_measure` is the unit (seconds, minutes, blocks etc).
By default, the value of `comparison_measure` is 'EQUAL'.
`set_comparison` specifies the behavior when the input_right or input_left return a list (e.g. from FILTERS).  Default is "ANY"; which means that if any of the comparisons are True, the comparator returns True.

```
ARGMAX = {"ordinal": 'FIRST' / <span>, "quantity": ATTRIBUTE}
ARGMIN = {"ordinal": 'FIRST' / <span>, "quantity": ATTRIBUTE}
```

```
NUM_BLOCKS = {"num_blocks": {"block_filters": {"has_x": span} }
             }
LINEAR_EXTENT = {"linear_extent" : {
                    "relative_direction"": "LEFT" / "RIGHT"/ "UP" / "DOWN"/ "FRONT" / "BACK"/ "AWAY" / "INSIDE" / "OUTSIDE", 
                    "frame": "SPEAKER" / "AGENT" / "ABSOLUTE" / {"player_span": span},
                    "has_measure" : span,
                    "source": REFERENCE_OBJECT,
		                "destination": REFERENCE_OBJECT
                    }
                }
```
Here LINEAR_EXTENT is used to mean the number of steps (in "has_measure" units, default is "blocks=meters") in the direction "relative_direction" from the "source" Location in the frame of reference of "frame".  If "source" and "destination" are specified, LINEAR_EXTENT evaluates to a number; otherwise, LINEAR_EXTENT evaluates to an ATTRIBUTE, a function that eats a (list of) memor(ies) and outputs a (list of) number(s)).  LINEAR_EXTENT can be used for "distance to" via relative_direction "AWAY".  "ABSOLUTE" is the coordinate system in which the agent starts.



Proposal:  add an extra "output_all" key to FILTERS, with possible values "ALL" or "RANDOM".  If the value is "ALL" returns a set of memories or values, and if "RANDOM" picks one uniformly at random. 

#### Conditions ####

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

#### EVENT draft ####
The "remove" key describes when the even should be deactivated/removed from memory; the "control" clause describes when the even tfires.  events are siblings of "action_sequence" and "dialogue_type"
```
EVENT = "event": {"action_sequence" : [ACTION, ...., ACTION],
                  "control": CONTROL,
                  "spatial_control" : SPATIAL_CONTROL
                  "remove": CONDITION}
```
