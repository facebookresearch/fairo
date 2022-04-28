## Input ##
At each step of the clarification process, the dashboard will pass the entire chat history to the NSP, in the following form.  Next to each user line is the expected result from the NSP and interpreter.

 - User:    [`command`]        ->  HUMAN_GIVE_COMMAND (resulting in a reference object clarification task)
 - Agent:   [`check_parse`]
 - User:    [`yes/n`o]         -> NOOP (caught by AwaitResponse, continue/end clarification task)
 - Agent:   [`point_at`]
 - User:    [`no`]             -> NOOP (caught by AwaitResponse, continue clarification task)
 - ...
 - Agent:   [`point_at`]
 - User:    [`yes`]            -> PUT_MEMORY (caught by AwaitResponse, end clarification task)

If the agent runs out of reference object candidates to point at, the clarification task ends on the final NOOP, no updates are made to memory and the user can proceed to label the command as resulting in a (presumably vision) error.

### NOOP Logical Form ###
<pre>
{
    'dialogue_type': 'NOOP'
}
</pre>

### PUT_MEMORY Logical Form Template ###
<pre>
{
    "dialogue_type": "PUT_MEMORY"
    "filters": {
        "output" : "MEMORY",
        "memory_type": "REFERENCE_OBJECT",
        "selector": {
            "ordinal": 1, 
        },
        "where_clause" : {
            "AND": {
                "pred_text": has_tag, 
                "obj_text": active_clarification
            }
        }
    },
    "upsert" : {
        "memory_data": {
            "memory_type": "TRIPLE",
            "triples": [{
                "pred_text": "has_tag",
                "obj_text": [ref_obj text span]
            }]
        } 
    }
}
</pre>