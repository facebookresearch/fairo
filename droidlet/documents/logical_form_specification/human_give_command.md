Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.


# Dialogue Types #

## Human Give Command Dialogue type ##
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : list(<CommandDict>)
}
```

## Actions ##

### Build Action ###
This is the action to Build a schematic at an optional location.

<pre>
{ "action_type" : BUILD,
  <a href="#location">Location</a>,
  <a href="#schematic">Schematic</a>,
}
</pre>

### Copy Action ###
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

<pre>
{ "action_type" : 'BUILD',
  <a href="#location">Location</a>,
  <a href="#reference_object">ReferenceObject</a>,
}
</pre>

### Spawn Action ###
This action indicates that the specified object should be spawned in the environment.
Spawn only has a name in the reference object.

<pre>
{ "action_type" : 'SPAWN',
  "reference_object" : FILTERS
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

```
{ "action_type" : 'FILL',
  <a href="#schematic">Schematic</a>,,
  <a href="#reference_object">ReferenceObject</a>,
}
```

#### Destroy ####
This action indicates the intent to destroy a block object.

<pre>
{ "action_type" : 'DESTROY',
    <a href="#reference_object">ReferenceObject</a>
}
</pre>

#### Move ####
This action states that the agent should move to the specified location.

```
{ "action_type" : 'MOVE',
  <a href="#location">Location</a>,
}
```

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
  <a href="#location">Location</a>,
  <a href="#schematic">Schematic</a>
}
</pre>


#### FreeBuild ####
This action represents that the agent should complete an already existing half-finished block object, using its mental model.

<pre>
{ "action_type" : 'FREEBUILD',
  <a href="#reference_object">ReferenceObject</a>,
  <a href="#location">Location</a>,
  "replace": True
}
</pre>

#### Dance ####
A movement where the sequence of poses or locations is determined, rather than just the final location.
Also has support for Point / Turn / Look.

<pre>
{ "action_type" : 'DANCE',
  <a href="#reference_object">ReferenceObject</a> (additional relative_direction values: ['CLOCKWISE', 'ANTICLOCKWISE']),
  DanceType
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
      <a href="#reference_object">ReferenceObject</a>,
    "receiver" : <a href="#reference_object">ReferenceObject</a> / <a href="#location">Location</a>
}
</pre>

#### Scout ####
This command expresses the intent to look for / find or scout something.

{
    "action_type" : 'SCOUT',
    <a href="#reference_object">ReferenceObject</a>,
    <a href="#location">Location</a>    
}


#### Modify action ####
Note that: This section is a temporary heristic based fix until we have a generative model that can handle "modify".

This command represents making a change or modifying a block object in a certain way.

[Examples](https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.o3lwuh3fj1lt)

Grammar proposal:
<pre>
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MODIFY',
      <a href="#reference_object">ReferenceObject</a>,
      "modify_dict" :  THICKEN/SCALE/RIGIDMOTION/REPLACE/FILL
    }]
}
</pre>
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
<pre>
<a id="location">"location": { </a>
          "text_span" : span,
          "steps" : span,
          "has_measure" : {"fixed_value" : text} / span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          <ReferenceObject>,
          }
 </pre>

Note: for "relative_direction" == 'BETWEEN' the location dict will have two children: 'reference_object_1' : <ReferenceObject> and 
'reference_object_2' : <ReferenceObject> in the sub-dictionary to represent the two distinct reference objects.

#### Reference Object ####
<pre>
  <a id="reference_object">"reference_object" : { </a>
      "special_reference" : {"fixed_value" : "SPEAKER" / "AGENT" / "SPEAKER_LOOK"} / {"coordinates_span" : span},
      "filters" : <FILTERS>
      }
  } 
</pre>
#### Remove Condition ####
```
"remove_condition" : {
      "condition_type" : 'ADJACENT_TO_BLOCK_TYPE' / 'NEVER',
      "block_type" : span
  }
```
#### Schematic ####

<pre>
<a id="schematic">"schematic": { </a>
    "text_span" : span,
    "filters" : <FILTERS>
}
</pre>

#### Repeat ####
<pre>
<a id="repeat">"repeat" : { </a>
  "repeat_key" : 'ALL'
}
</pre>

#### FACING ####
<pre>
{
  "text_span" : span,
  "yaw_pitch": span,
  "yaw": span,
  "pitch": span,
  "relative_yaw" = {"fixed_value": -360 / -180 / -135 / -90 / -45 / 45 / 90 / 135 / 180 / 360} / {"yaw_span": span},
  "relative_pitch" = {"fixed_value": -90 / -45 / 45 / 90} / {"pitch_span": span},
  "location": <LOCATION>
}
</pre>

#### DanceType ####
```
"dance_type" : {
  "filters": <FILTERS>,
  "point": <FACING>,
  "look_turn": <FACING>,
  "body_turn": <FACING>
}
```
