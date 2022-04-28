## Input ##
At each step of the clarification process, the dashboard will pass the entire chat history to the NSP, in the following form.  Next to each user line is the expected result from the NSP and interpreter.

[user] [command]        ->  HUMAN_GIVE_COMMAND (resulting in a reference object clarification task)
[agent] [check_parse]
[user] [yes/no]            -> NOOP (caught by AwaitResponse, continue/end clarification task)
[agent] [point_at]
[user] [no]             -> NOOP (caught by AwaitResponse, continue clarification task)
...
[agent] [point_at]
[user] [yes]             -> PUT_MEMORY (caught by AwaitResponse, end clarification task)

If the agent runs out of reference object candidates to point at, the clarification task ends on the final NOOP, no updates are made to memory and the user can proceed to label the command as resulting in a (presumably vision) error.

### NOOP Logical Form ###
<pre>
{'dialogue_type': 'NOOP'}
</pre>

### PUT_MEMORY Logical Form Template ###
<pre>
{
    "dialogue_type": "PUT_MEMORY"
    "filters": <FILTERS>,
    "upsert" : {
        "memory_data": {
            "memory_type": "TRIPLE",
            "triples": [{"pred_text": "has_tag", "obj_text": [ref_obj text span] }]
        } 
    }
}
</pre>

<FILTERS>  = {
    "output" : "MEMORY",
    "contains_coreference": "yes",
    "memory_type": "TASKS" / "REFERENCE_OBJECT" / "CHAT" / "PROGRAM" / "ALL",
    "selector": {
        "return_quantity": <ARGVAL> / span,
        "ordinal": {"fixed_value" : "FIRST"}, 
        "location":  <LOCATION>,
        "same":"REQUIRED"
    },
    "where_clause" : {
        "AND": [<COMPARATOR> / <TRIPLE>], 
    }
}

What is the role of filters in this case?  The interpreter already has access to the ref_obj memory.
What is the token that should go in the obj_text field?