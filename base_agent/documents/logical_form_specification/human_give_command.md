Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.


# Dialogue Types #

## Human Give Command Dialogue type ##
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : list(<CommandDict>)
}
```
The CommandDict for each action type is described in the [Action subsection](#actions).

## Actions ##

### Build Action ###
This is the action to Build a schematic at an optional location.

```
{ "action_type" : BUILD,
  <Location>,
  <Schematic>,
  <Repeat>,
  "replace" : True
}
```
Note also that: we can have an additional spatial_control: 'SURROUND' to replace `repeat_dir` when events and conditionals spec change is in.

### Copy Action ###
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

```
{ "action_type" : 'BUILD',
  <Location>,
  <ReferenceObject>,
  <Repeat>,
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
      <Repeat>,
      "triples": [{"pred_text": "has_name", "obj_text": {"fixed_value" : text} / span}]
    },
    <Repeat>,
    "replace": True
}
```

### Resume ###
This action indicates that the previous action should be resumed.

```
{ "action_type" : 'RESUME',
  "target_action_type": {"fixed_value" : text} / span
}
```

### Fill ###
This action states that a hole / negative shape needs to be filled up.

```
{ "action_type" : 'FILL',
  "triples": [{"pred_text": "has_block_type", "obj_text": {"fixed_value" : text} / span}]
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
  "target_action_type" : {"fixed_value" : text} / span
}
```

#### Stop ####
This action indicates stop.

```
{ "action_type" : 'STOP',
  "target_action_type": {"fixed_value" : text} / span
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
    <Repeat>,
    "triples": [{"pred_text": "has_x", "obj_text": {"fixed_value" : text} / span}]
    },
  <StopCondition>,
  <Repeat>,
  "replace": True  
}
```
where `has_x` can be : `has_size`, `has_length`, `has_depth`, `has_width`

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
  <Repeat>,
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
          "has_measure" : {"fixed_value" : text} / span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          <ReferenceObject>,
          }
 ```

Note: for "relative_direction" == 'BETWEEN' the location dict will have two children: 'reference_object_1' : <ReferenceObject> and 
'reference_object_2' : <ReferenceObject> in the sub-dictionary to represent the two distinct reference objects.

#### Reference Object ####
```
"reference_object" : {
      "text_span" : span,
      <Repeat>,
      "special_reference" : {"fixed_value" : "SPEAKER" / "AGENT" / "SPEAKER_LOOK"} / {"coordinates_span" : span},
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
          <Repeat>, 
          "triples" : [{"pred_text": "has_x", "obj_text": {"fixed_value" : text} / span}]
      }
```
where `has_x` can be : `has_block_type`, `has_name`, `has_size`, `has_orientation`, `has_thickness`, `has_colour`, `has_height`, `has_length`, `has_radius`, `has_slope`, `has_width`, `has_base`, `has_depth`, `has_distance` 
`schematic` can have an additional 'SURROUND' spatial_control for when we introduce `spatial_control` once the events and conditonals spec change is in.

#### Repeat ####
Loop over until a count (previously Repeat for):
```
"stop_condition" : {
  "condition_type" : "COMPARATOR",
  "condition" : {
    "comparison_type" : "EQUAL",
    "input_left": {
      "filters" : {
        "output" : {"attribute" : "RUN_COUNT"},
        "memory_type" : "_THIS"
      }
    },
    "input_right" : {
      "value" : span
    }
  }
}
```
Here the `input_left` field returns the `RUN_COUNT` attribute of the task that the loop is being called on, and that is compared against the `value` field in `input_right` (which will be the loop count). Note that `"memory_type": "_THIS"` denotes the current task.

Repeat `ALL` will be :
```
"repeat" : {"repeat_key" : 'ALL'}
```
for now. But this will be changed in a later iteration.

Note also that: `"repeat_dir"` from before will now sit in `spatial_control` - this change should go in once the events and conditionals spec change is in.

#### FACING ####
```
{
  "text_span" : span,
  "yaw_pitch": span,
  "yaw": span,
  "pitch": span,
  "relative_yaw" = {"fixed_value": -360 / -180 / -135 / -90 / -45 / 45 / 90 / 135 / 180 / 360} / {"yaw_span": span},
  "relative_pitch" = {"fixed_value": -90 / -45 / 45 / 90} / {"pitch_span": span},
  "location": <LOCATION>
}
```

#### DanceType ####
```
"dance_type" : {
  "dance_type_name": {"fixed_value" : text} / span,
  "dance_type_tag": span,
  "point": <FACING>,
  "look_turn": <FACING>,
  "body_turn": <FACING>
}
```
