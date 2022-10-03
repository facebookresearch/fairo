
# Dialogue Types #

### Human Give Command Dialogue type ###
<pre>
{ 
  "dialogue_type": "HUMAN_GIVE_COMMAND",
  <a href="#event">&ltEVENT&gt</a>
}
</pre>

### Get Memory Dialogue Type ###
<pre>
{
  "dialogue_type": "GET_MEMORY",
  "filters": <a href="#filters">&ltFILTERS&gt</a>
}
</pre>

### Put Memory Dialogue Type ###
<pre>
{
  "filters": <a href="#filters">&ltFILTERS&gt</a>,
  "upsert" : {
      "memory_data": {
        "memory_type": "REWARD" / "TRIPLE",
        "reward_value": "POSITIVE" / "NEGATIVE",
        "triples": [{"pred_text": "has_x", "obj_text": {"fixed_value" : text} / span}]
      } 
   }
}
</pre>
where `has_x` is one of : `has_tag`, `has_colour`, `has_size`.  The value of the "filters" key tells the agent which memory to modify or tag.

### Noop Dialogue Type ###
```
{ "dialogue_type": "NOOP"}
```
# Events and Actions #

<pre>
<a id="event">EVENT</a> = { 
  "event_sequence": [<a href="#event">&ltEVENT&gt</a>/<a href="#action">&ltACTION&gt</a>, …, <a href="#event">&ltEVENT&gt</a>/<a href="#action">&ltACTION&gt</a>],
  "init_condition": <a href="#condition">&ltCONDITION&gt</a>,
  "terminate_condition": <a href="#condition">&ltCONDITION&gt</a>,
  "spatial_control": SPATIAL_CONTROL
}
</pre>

Each condition key and the "spatial_control" blocks are optional. The default (if a key is not set) for "init_condition" is {"fixed_value":  "ALWAYS"}.  The default for "terminate_condition"  is THIS_SEQUENCE_1_TIMES_CONDITION , which uses a special value for use only inside a <a href="#condition">CONDITION</a> in an <a href="#event">EVENT</a> block: "THIS", referring to the value of the "event_sequence" key that is a sibling of the "\*condition" key inside an <a href="#event">EVENT</a> block.  


## Actions ##
<a id="action"> Actions </a> are an "atomic" <a href="#event">EVENT</a>, and can be distinguished by the "action_type" key.

### Say Action ###
This is the action that triggers the agent to say or speak something (represented by `say_span`):
<pre>
{"action_type" : 'SAY',
 "say_span" : span
}
</pre>


### Build Action ###
This is the action to Build a schematic at an optional location.

<pre>
{"action_type" : BUILD,
  <a href="#location">&ltLOCATION&gt</a>,
  <a href="#schematic">&ltSCHEMATIC&gt</a>
}
</pre>

### Copy Action ###
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

<pre>
{"action_type" : 'BUILD',
  <a href="#location">&ltLOCATION&gt</a>,
  <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>
}
</pre>

### Spawn Action ###
This action indicates that the specified object should be spawned in the environment.
Spawn only has a name in the reference object.

<pre>
{ "action_type" : 'SPAWN',
  "reference_object" : <a href="#filters">&ltFILTERS&gt</a>
}
</pre>

### Resume ###
This action indicates that the previous action should be resumed.

<pre>
{ "action_type" : 'RESUME',
  "target_action_type": {"fixed_value" : text} / span
}
</pre>

### Fill ###
This action states that a hole / negative shape needs to be filled up.

<pre>
{ "action_type" : 'FILL',
  <a href="#schematic">&ltSCHEMATIC&gt</a>,
  <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>
}
</pre>

#### Destroy ####
This action indicates the intent to destroy a block object.

<pre>
{ "action_type" : 'DESTROY',
    <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>
}
</pre>

#### Move ####
This action states that the agent should move to the specified location.

<pre>
{ "action_type" : 'MOVE',
  <a href="#location">&ltLOCATION&gt</a>
}
</pre>

#### Undo ####
This action states the intent to revert the specified action, if any.

<pre>
{ "action_type" : 'UNDO',
  "target_action_type" : {"fixed_value" : text} / span
}
</pre>

#### Stop ####
This action indicates stop.

<pre>
{ "action_type" : 'STOP',
  "target_action_type": {"fixed_value" : text} / span
}
</pre>

#### Dig ####
This action represents the intent to dig a hole / negative shape of optional dimensions at an optional location.
The `Schematic` child in this only has a subset of properties.

<pre>
{ "action_type" : 'DIG',
  <a href="#location">&ltLOCATION&gt</a>,
  <a href="#schematic">&ltSCHEMATIC&gt</a>
}
</pre>


#### FreeBuild ####
This action represents that the agent should complete an already existing half-finished block object, using its mental model.

<pre>
{ "action_type" : 'FREEBUILD',
  <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>,
  <a href="#location">&ltLOCATION&gt</a>
}
</pre>

#### Dance ####
A movement where the sequence of poses or locations is determined, rather than just the final location.
Also has support for Point / Turn / Look.

<pre>
{ "action_type" : 'DANCE',
  <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a> (additional relative_direction values: ['CLOCKWISE', 'ANTICLOCKWISE']),
  <a href="#dance_type">&ltDANCE_TYPE&gt</a>
}
</pre>

#### Get ####
The GET action_type covers the intents: bring, get and give.

The Get intent represents getting or picking up something. This might involve first going to that thing and then picking it up in bot’s hand. The receiver here can either be the agent or the speaker (or another player).
The Give intent represents giving something, in Minecraft this would mean removing something from the inventory of the bot and adding it to the inventory of the speaker / other player.
The Bring intent represents bringing a reference_object to the speaker or to a specified location.

<pre>
{
    "action_type" : 'GET',
    <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>,
    "receiver" : <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a> / <a href="#location">&ltLOCATION&gt</a>
}
</pre>

#### Scout ####
This command expresses the intent to look for / find or scout something.

<pre>
{
    "action_type" : 'SCOUT',
    <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>,
    <a href="#location">&ltLOCATION&gt</a>    
}
</pre>


# Subcomponents of logical forms #

## CONDITION ##
<pre>
<a id="condition"> CONDITION = { </a>
  "condition": <a href="#comparator">&ltCOMPARATOR&gt</a>/<a href="#time_condition">&ltTIME_CONDITION&gt</a>/<a href="#and_condition">&ltAND_CONDITION&gt</a>/<a href="#or_condition">&ltOR_CONDITION&gt</a>/<a href="#not_condition">&ltNOT_CONDITION&gt</a>,
  "condition_span": span}
</pre>
AND OR and NOT modify other conditions:
<pre>
 <a id="and_condition">AND_CONDITION</a> = {"and_condition": [<a href="#condition">&ltCONDITION&gt</a>, ..., <a href="#condition">&ltCONDITION&gt</a>]} 
</pre>
<pre>
 <a id="or_condition">OR_CONDITION</a> = {"or_condition": [<a href="#condition">&ltCONDITION&gt</a>, ..., <a href="#condition">&ltCONDITION&gt</a>]} 
</pre>
<pre>
 <a id="not_condition">NOT_CONDITION</a> = {"not_condition": <a href="#condition">&ltCONDITION&gt</a>} 
</pre>
<pre>
<a id="time_condition">TIME_CONDITION</a> = {
    "comparator": <a href="#comparator">&ltCOMPARATOR&gt</a>,
    "special_time_event" : 'SUNSET / SUNRISE / DAY / NIGHT / RAIN / SUNNY',
    "event": <a href="#condition">&ltCONDITION&gt</a>,
  }
</pre>
Note that `event` in time_condition represents when to start the timer and
`input_left` by default in time condition marks the time since the event condition.


#### Location ####
<pre>
<a id="location">"location": { </a>
          "text_span" : span,
          "steps" : span,
          "has_measure" : {"fixed_value" : text} / span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
           <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>,
          }
 </pre>

Note: for "relative_direction" == 'BETWEEN' the location dict will have two children: 'reference_object_1' : <a href="#reference_object"> \<REFERENCE_OBJECT\></a>, and 
'reference_object_2' : <a href="#reference_object">\<REFERENCE_OBJECT\></a>, in the sub-dictionary to represent the two distinct reference objects.

#### Reference Object ####
<pre>
  <a id="reference_object">"reference_object" : { </a>
      "special_reference" : {"fixed_value" : "SPEAKER" / "AGENT" / "SPEAKER_LOOK"} / {"coordinates_span" : span},
      "filters" : <a href="#filters">&ltFILTERS&gt</a>
      }
  } 
</pre>

#### Schematic ####

<pre>
<a id="schematic">"schematic": { </a>
    "text_span" : span,
    "filters" : <a href="#filters">&ltFILTERS&gt</a>
}
</pre>

#### FACING ####
<pre>
<a id="facing">"facing" : { </a>
  "text_span" : span,
  "yaw_pitch": span,
  "yaw": span,
  "pitch": span,
  "relative_yaw" = {"fixed_value": -360 / -180 / -135 / -90 / -45 / 45 / 90 / 135 / 180 / 360} / {"yaw_span": span},
  "relative_pitch" = {"fixed_value": -90 / -45 / 45 / 90} / {"pitch_span": span},
  "location": <a href="#location">&ltLOCATION&gt</a>
}
</pre>

The specificatition sets positive yaw to be rotations to the left and negative yaw to be rotations to the right;
and positive pitch up (90) and negative pitch down (-90).  TODO switch all angular fixed values to radians.
	
#### DanceType ####
<pre>
<a id="dance_type"/> : { 
  "filters": <a href="#filters">&ltFILTERS&gt</a>,
  "point": <a href="#facing">&ltFACING&gt</a>,
  "look_turn": <a href="#facing">&ltFACING&gt</a>,
  "body_turn": <a href="#facing">&ltFACING&gt</a>
}
</pre>


# Filters #

This is a query language for accessing the agent's memory.  Records are stored as memory nodes.  
The supported fields are : 
- `output` : This field specifies the expected output type. In the `output` field we can either have a `"memory"` value (which specifies returning the memory node), `"count"` that represents the equivalent of `count(*)` (for eg: "how many houses on your left") or `attribute` representing a generalized column of the memory node (equivalent to `Select <attribute> from memory_node`) (for example: "what colour is the cube in front of me"). 
- `contains coreference` : This field specifies whether the query has coreference that needs to be resolved based on dialogue context.
- `memory_type` : Specifies the type of memory is the name of memory node. The default value for this is : `"REFERENCE_OBJECT"`.
- `where_clause`: This field describes the properties that memory nodes must have to be returned.
- `selector` : Specifies how to sub-select from the memory nodes that satisfy the where_clause; perhaps by comparing amongst the returned records
  - `return_quantity` use an attribute, perhaps with sorting or ranking, to determine the records to be returned.
  - `ordinal` which specifies the number of entries to be returned.
  - `same` that specifies whether we are allowed to return the exact copies of an object. 
  - `location` : <a href="#location">location</a> special shortcut for selections based on spatial location.

<pre>
<a id="filters">&ltFILTERS&gt </a> = {
      "output" : "MEMORY" / "COUNT" / <a href="#attribute">&ltATTRIBUTE&gt</a>,
      "contains_coreference": "yes",
      "memory_type": "TASKS" / "REFERENCE_OBJECT" / "CHAT" / "PROGRAM" / "ALL",
      "selector": {
        "return_quantity":<a href="#argval"> &ltARGVAL&gt</a> / "RANDOM" / "ALL" / span,
        "ordinal": {"fixed_value" : "FIRST"} / <span>, 
        "location":  <a href="#location">&ltLOCATION&gt</a>,
        "same":"ALLOWED"/"DISALLOWED"/"REQUIRED"
      },
      "where_clause" : {
        "AND": [<a href="#comparator">&ltCOMPARATOR&gt</a> / <a href="#triple">&ltTRIPLE&gt</a>], 
        "OR": [<a href="#comparator">&ltCOMPARATOR&gt</a> / <a href="#triple">&ltTRIPLE&gt</a>], 
        "NOT": [<a href="#comparator">&ltCOMPARATOR&gt</a> / <a href="#triple">&ltTRIPLE&gt</a>]
        }
      }
</pre>
## Where clause ##
The value of this key is analagous to the `WHERE` clause in a SQL query. The where clause allows these three keys:
"AND" , "OR" and "NOT".
The values for these keys are a lit of dictionaries where each dictionary is either: a 
<a href="#comparator">\<COMPARATOR\></a> or <a href="#triple"> \<TRIPLE\></a>.

## ATTRIBUTE ##

An ATTRIBUTE specifies a function that takes a memory node as input and outputs a value:

<pre>
{
<a id="attribute"> "attribute" </a> : 'HEIGHT' / 'WIDTH' / 'X' / 'Y' / 'Z' / 'REF_TYPE' / 
               'HEAD_PITCH' / 'HEAD_YAW' / 'BODY_YAW'/ 'NAME' / 
               'BORN_TIME' / 'MODIFY_TIME' / 'VISIT_TIME' / 'FINISHED_TIME' /
               'SPEAKER' / 'CHAT' / 'LOGICAL_FORM' /  'NAME' / 
               'SIZE' / 'COLOUR' / 'LOCATION' /'TAG' / <a href="#num_blocks">&ltNUM_BLOCKS&gt</a> /   
	       <a href="#linear_extent">&ltLINEAR_EXTENT&gt</a> / {"task_info" : {"reference_object" : <a href="#attribute">&ltATTRIBUTE&gt</a> }},
 "at_time": <a href="#time_condition">&ltTIME_CONDITION&gt</a>
}
</pre>
The default of the "at_time" key (if it is ommitted) is the time when the query is issued.

### NUM BLOCKS ###
This represents number of blocks. For example: "go to the house with most red blocks":
<pre>
{<a id="num_blocks"> "num_blocks" </a> :  {
    "block_filters":  <a href="#filters">&ltFILTERS&gt</a>}
}
</pre>

### LINEAR EXTENT ###
This is used to mean the number of steps (in "has_measure" units, default is "blocks=meters") in the direction "relative_direction" from the "source" location in the frame of reference of "frame".  If "source" and "destination" are specified, LINEAR_EXTENT evaluates to a number; otherwise, LINEAR_EXTENT evaluates to an <a href="#ttribute">\<ATTRIBUTE\></a>, a function that takes a (list of) memor(ies) and outputs a (list of) number(s)).  LINEAR_EXTENT can be used for "distance to" via relative_direction "AWAY".  "ABSOLUTE" is the coordinate system in which the agent starts.
<pre>
{
<a id="linear_extent"> "linear_extent" : { </a>
    "relative_direction": "LEFT" / "RIGHT"/ "UP" / "DOWN"/ "FRONT" 
                          / "BACK"/ "AWAY" / "INSIDE" / "OUTSIDE", 
    "frame": {"fixed_value": "SPEAKER" / "AGENT" / "ABSOLUTE"} / {"player_span": span},
    "has_measure" : {"fixed_value" : text} / span,
    "source": <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a> ,
    "destination": <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a> }
    }
}
</pre>

### ARGVAL ###
This defines either: 
- the polarity of maximum, so ranking from max to min on the quantity or
- the polarity of minimum, so ranking from min to max on the quantity
<pre>
{<a id="argval"> "argval"</a> : {
    "polarity" : "MAX" / "MIN", 
    "quantity": <a href="#attribute"> &ltATTRIBUTE&gt </a>
    }
}
</pre>


### COMPARATOR ###
Comparator compares two values.
- `comparison_type` represents the kind of comparison (>, <, >=, != , =)
- `input_left` can be either a <a href="#filters"> \<FILTERS\></a> , a span, or an <a href="#attribute">\<ATTRIBUTE\></a>; `input_right` can be either a <a href="#filter">\<FILTERS\> </a> or a span (but not an <a href="#attribute">\<ATTRIBUTE\></a>).   
- `comparison_measure` is the unit (seconds, minutes, blocks etc).
- `set_comparison` specifies the behavior when the input_right or input_left return a list (e.g. from <a href="#filters">\<FILTERS\></a>).  Default is `"ANY"`; which means that if any of the comparisons are True, the comparator returns True.
- <a href="#attribute"> \<ATTRIBUTE\> </a> in `input_left` is used when the comparator is a WHERE clause; and the <a href="#attribute">\<ATTRIBUTE\></a> applied to a record is compared against the `input_right` to decide if the record is accepted by the clause.  When the comparator is used in a <a href="#condition">\<CONDITION\></a>, both `input_left` and `input_right` are "fixed" ( <a href="#filters">\<FILTERS\></a> or a span)
		
<pre>
<a id="comparator"> COMPARATOR =  </a>

{
  "input_left" : <a href="#filters">FILTERS</a> / <a href="#attribute">Attribute</a> / {"fixed_value" : text} / span 
  "comparison_type" : "GREATER_THAN" / "LESS_THAN" / "GREATER_THAN_EQUAL" / 
                      "LESS_THAN_EQUAL" / "NOT_EQUAL" / "EQUAL" / <CLOSE_TO> / <MOD_EQUAL> / <MOD_CLOSE>,
  "input_right" : <a href="#filters">FILTERS</a> / <a href="#attribute">Attribute</a> / {"fixed_value" : text} /span 
  "comparison_measure" : {"fixed_value" : text} / span,
  "set_comparison": "ANY"/ "ALL"
}
</pre>

#### CLOSE_TO ####
This defines a tolerance of proximity :
```
{"close_tolerance": {"fixed_value" : "DEFAULT"} / span}
```

#### MOD_CLOSE ####
This defines a modulus with a `close_tolerance`:
```
{ "modulus": {"fixed_value" : "DEFAULT"} / span, 
  "close_tolerance": {"fixed_value" : "DEFAULT"} / span}
```

#### MOD_EQUAL ####
`MOD_EQUAL` in COMPARATOR is for e.g. handling time conditions like 'every 5 seconds' :
```
{ "modulus": {"fixed_value" : "DEFAULT"} / span}
```


## TRIPLES ##
corresponds to a (subject, predicate, object) triple.
Each of these keys can be one of:
- `subj` or `subj_text`
- `pred` or `pred_text`
- `obj` or `obj_text`
In e.g. `obj` vs. `obj_text`, the former would be used to fix an explicit memid, and the latter to refer to the text associated to a NamedAbstraction 
<pre>
<a id="triple"> TRIPLE =  </a>
{
  "pred_text" / "pred": "has_x", 
  "obj_text" / "obj" : {"fixed_value" : text} / span / <a href="#filters">FILTERS</a>, 
  "subj_text" / "subj" : {"fixed_value" : text} / span / <a href="#filters">FILTERS</a>
 }
</pre>
Note:
- `AGENT` is denoted by the triple:
```
{
  "pred_text": "has_tag", 
  "obj_text": {"fixed_value" : "SELF"} }
```
and `'memory_type': 'REFERENCE_OBJECT'` 

- `SPEAKER` is denoted by the triple:
```
{
  "pred_text": "has_tag", 
  "obj_text": {"fixed_value" : "SPEAKER"}}
```
and `'memory_type': 'REFERENCE_OBJECT'` 

- Current task is : `'memory_type': 'TASKS'` and 
```
{"pred_text": "has_tag", 
 "obj_text": {"fixed_value" : "CURRENTLY_RUNNING"}}
``` 
- A completed task by: `'memory_type': 'TASKS'` and 
```
{"pred_text": "has_tag", 
 "obj_text": {"fixed_value" : "FINISHED"}}
``` 

- Task name is represented using :
```
{"pred_text": "has_name", 
 "obj_text": {
   "fixed_value": "BUILD" / "DIG" / "FILL" / "SPAWN" / 
                  "RESUME" / "FILL" / "DESTROY" / "MOVE" / 
                  "DIG" / "GET" / "DANCE" / "FREEBUILD" /
                  "STOP" / "UNDO"}
}
```


### Section for TODOs and proposals ###
Proposal:  Add an extra "output_all" key to FILTERS, with possible values "ALL" or "RANDOM".  If the value is "ALL" returns a set of memories or values, and if "RANDOM" picks one uniformly at random. 


TODO: Add "in_agent_view" like field to distinguish between "if you see a pig" vs "if there is a pig"





#### Modify action ####
Note that: This section is a temporary heristic based fix until we have a generative model that can handle "modify".

This command represents making a change or modifying a block object in a certain way.

[Examples](https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.o3lwuh3fj1lt)

Grammar proposal:
<pre>
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MODIFY',
      <a href="#reference_object">&ltREFERENCE_OBJECT&gt</a>,
      "modify_dict" :  THICKEN/SCALE/RIGIDMOTION/REPLACE/FILL
    }]
}
</pre>
Where
```
THICKEN = {"modify_type": "THICKER"/"THINNER", "num_blocks": span}
SCALE = {"modify_type": "SCALE", "text_scale_factor": span, "categorical_scale_factor": "WIDER"/"NARROWER"/"TALLER"/"SHORTER"/"SKINNIER"/"FATTER"/"BIGGER"/"SMALLER"}
RIGIDMOTION = {"modify_type": "RIGIDMOTION", "categorical_angle": "LEFT"/"RIGHT"/"AROUND", "MIRROR"=True/False, "location": <a href="#location">&ltLOCATION&gt</a>}
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
