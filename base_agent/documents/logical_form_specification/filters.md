Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# Filters #

Filters helps us specify reference objects and extend Embodied Question Answering pipeline. Filters are used inside `reference_object` and dialogue types: `GET_MEMORY` and `PUT_MEMORY`.
The supported fields are : 
- `output` : This field specifies the expected output type. In the `output` field we can either have a `"memory"` value which means return the entire memory node (for eg: "is that red"), or `"count"` that represents the equivalent of `count(*)` (for eg: "how many houses on your left") or `attribute` representing the column name of that memory node (equivalent to `Select <attribute> from memory_node`) (for example: "what colour is the cube in front of me"). 
- `contains coreference` : This field specifies whether the memory node has a coreference that needs to be resolved based on dialogue context.
- `memory_type` : Specifies the type of memory is the name of memory node. The default value for this is : `"REFERENCE_OBJECT"`.
- `argval` : This field specifies that is there is some form of ranking.
- `comparator`: This field specifies some form of comparison between two values.
- `triples` : This is a list of triple dictionaries that have a subject, predicate and object.
- `author` : Specifies the author of the change. For eg in "go to the house I made last", author is "SPEAKER" here.
- `location` : location of the object and is defined [here](human_give_command.md#location)


```
FILTERS = { 
      "output" : "MEMORY" / "COUNT" / <ATTRIBUTE>,
      "contains_coreference": "yes",
      "memory_type": "TASKS" / "REFERENCE_OBJECT" / "CHAT" / "PROGRAM" / "ALL",
      "argval" : <ARGMAX> / <ARGMIN> ,
      "comparator": [<COMPARATOR> , ...],
      "triples": <TRIPLES>,
      "author":  {"fixed_value" : "AGENT" / "SPEAKER"} / span,
      "location": <LOCATION> }
```

## ATTRIBUTE ##

This defines attributes of the memory node that can be exposed as outputs and the representation is as follows:

```
{"attribute" : 'HEIGHT' / 'WIDTH' / 'X' / 'Y' / 'Z' / 'REF_TYPE' / 
               'HEAD_PITCH' / 'HEAD_YAW' / 'BODY_YAW'/ 'NAME' / 
               'BORN_TIME' / 'MODIFY_TIME' / 'SPEAKER' / 'VISIT_TIME' /  
               'FINISHED_TIME' / 'CHAT' / 'LOGICAL_FORM' /  'NAME' / 
               'SIZE' / 'COLOUR' / 'LOCATION' /'TAG' / <NUM_BLOCKS> /   <LINEAR_EXTENT> / {"task_info" : {"reference_object" : <ATTRIBUTE>}}
}
```

### NUM BLOCKS ###
This represents number of blocks and hence a filter over those. For example: "go to the house with most red blocks". 
Representation:
```
{"num_blocks": {
                "block_filters": {
                    "triples" : <TRIPLES> } }
}
```

### LINEAR EXTENT ###
This is used to mean the number of steps (in "has_measure" units, default is "blocks=meters") in the direction "relative_direction" from the "source" location in the frame of reference of "frame".  If "source" and "destination" are specified, LINEAR_EXTENT evaluates to a number; otherwise, LINEAR_EXTENT evaluates to an <ATTRIBUTE>, a function that takes a (list of) memor(ies) and outputs a (list of) number(s)).  LINEAR_EXTENT can be used for "distance to" via relative_direction "AWAY".  "ABSOLUTE" is the coordinate system in which the agent starts.
```
{"linear_extent" : {
            "relative_direction": "LEFT" / "RIGHT"/ "UP" / "DOWN"/ "FRONT" 
                                  / "BACK"/ "AWAY" / "INSIDE" / "OUTSIDE", 
            "frame": {"fixed_value": "SPEAKER" / "AGENT" / "ABSOLUTE"} / {"player_span": span},
            "has_measure" : {"fixed_value" : text} / span,
            "source": <REFERENCE_OBJECT>,
		        "destination": <REFERENCE_OBJECT> }
            }
}
```

### ARGMAX ###
This defines the polarity of maximum, so ranking from max to min on the quantity: 
```
{ "polarity" : "MAX", 
  "ordinal": {"fixed_value" : "FIRST"} / <span>, 
  "quantity": <ATTRIBUTE>
}
```

### ARGMIN ###
This defines the polarity of minimum, so ranking from min to max on the quantity :
```
{ "polarity" : "MIN", 
  "ordinal": {"fixed_value" : "FIRST"} / <span>, 
  "quantity": <ATTRIBUTE>
}
```
 
### COMPARATOR ###
Comparator compares two values on the left and right based on the comparison type.
- `comparison_type` represents the kind of comparison (>, <, >=, != , =)
- `input_left` is the candidate on left of the comparison,`input_right` is the candidate on right of the comparison,
- `input_left` or `input_right` can be either a \<FILTER\>, a span, or an \<ATTRIBUTE\>.   \<ATTRIBUTE\> is used when the comparator is part of a \<FILTER\>; and the \<ATTRIBUTE\> applied to a list of memories is used for making a decision on their inclusion.
- `comparison_measure` is the unit (seconds, minutes, blocks etc).
- `set_comparison` specifies the behavior when the input_right or input_left return a list (e.g. from \<FILTERS\>).  Default is `"ANY"`; which means that if any of the comparisons are True, the comparator returns True.
		
```
{
  "input_left" : {"value_extractor" : <FILTERS> / <ATTRIBUTE> / {"fixed_value" : text} / span }
  "comparison_type" : "GREATER_THAN" / "LESS_THAN" / "GREATER_THAN_EQUAL" / 
                      "LESS_THAN_EQUAL" / "NOT_EQUAL" / "EQUAL" / <CLOSE_TO> / <MOD_EQUAL> / <MOD_CLOSE>,
  "input_right" : {"value_extractor" : <FILTERS> / <ATTRIBUTE> / {"fixed_value" : text} /span }
  "comparison_measure" : {"fixed_value" : text} / span,
  "set_comparison": "ANY"/ "ALL"
}
```

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
This is a list of triples, where each triple is a dictionary with optional : subject , predicate and object keys.
Each of these keys can be one of:
- `subj` or `subj_text`
- `pred` or `pred_text`
- `obj` or `obj_text`

```
[
  {"pred_text" / "pred": "has_x", 
  "obj_text" / "obj" : {"fixed_value" : text} / span / <FILTERS>, 
  "subj_text" / "subj" : {"fixed_value" : text} / span / <FILTERS>},
  ....
]
```
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
 "obj_text": {"fixed_value": "BUILD" / "DIG" / "FILL" / "SPAWN" / 
                             "RESUME" / "FILL" / "DESTROY" / "MOVE" / 
                             "DIG" / "GET" / "DANCE" / "FREEBUILD" /
                             "STOP" / "UNDO"}
}
```


### Section for TODOs and proposals ###
Proposal:  Add an extra "output_all" key to FILTERS, with possible values "ALL" or "RANDOM".  If the value is "ALL" returns a set of memories or values, and if "RANDOM" picks one uniformly at random. 


TODO: Add "in_agent_view" like field to distinguish between "if you see a pig" vs "if there is a pig"
