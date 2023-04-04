
# Dialogue Types #
These describe the DSL for dialog generation *from the agent to the user*.  For the DSL describing dialog commands coming from the user into the agent, 
see [`agent_dsl.md`](https://github.com/facebookresearch/fairo/blob/main/droidlet/documents/logical_form_specification/agent_dsl.md)

## Clarification Dialogue Type ##
<pre>
{ 
  "dialogue_type": "CLARIFICATION",
  "action": <a href="https://github.com/facebookresearch/fairo/blob/main/droidlet/documents/logical_form_specification/agent_dsl.md#actions">&ltACTION&gt</a>,
  "class": <a href="#class">&ltCLASS&gt</a>
}
</pre>

The ACTION field is the atomic action branch of the original (HUMAN_GIVE_COMMAND) logical form, 
the interpretation of which triggered a clarification task.

### Clarification Classes ###
<a id="class"> Clarification Classes </a> represent the possible currently supported types of clarification task.  Intuitively, they represent the 
nature of the error encounted when the interpreter attempted to resolve the command logical form into a set of actions.  
For example, when attempting to retrieve the memory of a `reference_object`, whether not enough or too many matches were returned.

#### No Matching Reference Objects In Memory ####
The situation in which one or more reference objects were contained in the original logical form, but no matches were found in memory.
The agent will clarify using all reference objects in the local scene.
<pre>
"class": {
          "error_type" : 'REF_NO_MATCH',
          "candidates": [<a href="#candidate">&ltCANDIDATE&gt</a>, <a href="#candidate">&ltCANDIDATE&gt</a>, ... <a href="#candidate">&ltCANDIDATE&gt</a>]
          }
</pre>

#### Too Few Reference Objects In Memory ####
The situation in which more than one reference object was contained in the original command logical form, but fewer than the requisite number
were found in memory.  The agent will clarify using all reference objects in the local scene, but will not clarify the exact match(es).
<pre>
"class": {
          "error_type" : 'REF_TOO_FEW',
          "candidates": [<a href="#candidate">&ltCANDIDATE&gt</a>, <a href="#candidate">&ltCANDIDATE&gt</a>, ... <a href="#candidate">&ltCANDIDATE&gt</a>]
          }
</pre>

#### Too Many Reference Objects In Memory ####
The situation in which more reference objects were found in memory than were contained in the original command logical form.  The agent will 
clarify using the original set of matches.
<pre>
"class": {
          "error_type" : 'REF_TOO_MANY',
          "candidates": [<a href="#candidate">&ltCANDIDATE&gt</a>, <a href="#candidate">&ltCANDIDATE&gt</a>, ... <a href="#candidate">&ltCANDIDATE&gt</a>]
          }
</pre>

#### Bad Parse ####
*Future TODO*
The situation in which the original parse is obviously bad, eg. invalid JSON or an impossible tree.
<pre>
"class": {
          "error_type" : 'BAD_PARSE',
          ...?
          }
</pre>

## Subcomponents of Logical Forms ##

### Reference Object Candidates ###
A <a id="candidate">CANDIDATE</a> is a possible match for an ambiguous reference object.
<pre>
"candidate": {
          "node_type" : 'VoxelObject / BlockObject / Mob / InstSeg',
          "speaker_distance": float, 
          "most_recent_update": int,
          "who_placed": 'PlayerNode name / Self / None'
          }
</pre>


