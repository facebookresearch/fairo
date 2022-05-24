# Clarification Templates #

The following are templated interactions meant to show how the NSP should respond in different clarification cases

## Basic Clarification ##

### Basic Input ###

At each step of the clarification process, the dashboard will pass the entire chat history to the NSP, in the following form.  Next to each user line is the expected result from the NSP and interpreter.

- User:    [`command`]        ->  HUMAN_GIVE_COMMAND (resulting in a reference object clarification task)
- Agent:   [`check_parse`]
- User:    [`yes/no`]         -> NOOP (caught by AwaitResponse, continue/end clarification task)
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
        "where_clause" : {
            "AND": {
                "pred_text": "has_tag",
                "obj_text": "active_clarification"
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

## Clarification Follow-Up ##

### Follow-Up Input ###

The user is given the opportunity to interrupt the clarification process with a second chat that is meant to either modify or replace the first.  For the purposes of this template it is assumed that this will happen right away (before answering the `check_parse` question), but in theory it should work anywhere as long as the chat history is available.  It is also assumed here that the follow up chat results in a new logical form that can be executed without clarification, but again, there is no reason that further clarification is prohibited.

- User:    [`command`]        -> HUMAN_GIVE_COMMAND (resulting in a reference object clarification task)
- Agent:   [`check_parse`]
- User:    [`follow-up`}      -> HUMAN_GIVE_COMMAND (resulting in a new logical form, AwaitResponse ends clarification)

### HUMAN_GIVE_COMMAND Follow Up Example ###

#### Initial Command and LF ###

"destory the cube"
<pre>
{
    'dialogue_type': 'HUMAN_GIVE_COMMAND',
    'event_sequence': [{
        'action_type': 'DESTROY',
        'reference_object': {
            'filters': {
                'where_clause': {
                    'AND': [{
                        'pred_text': 'has_name',
                        'obj_text': [0,[2,2]]
                    }]
                }
            },
            'text_span': 'cube'
        }
    }]
}
</pre>

#### Modification Follow Up Chat History and LF ###

"User: destory the cube Agent: I'm not sure about something. I think you wanted me to destroy a cube, is that right? User: the one next to the hole"
<pre>
{
    'dialogue_type': 'HUMAN_GIVE_COMMAND',
    'event_sequence': [{
        'action_type': 'DESTROY',
        'reference_object': {
            'filters': {
                'where_clause': {
                    'AND': [{
                        'pred_text': 'has_name',
                        'obj_text': [0,[3,3]]
                    }]
                },
                'selector': {
                    'location': {
                        'reference_object': {
                            'filters': {
                                'where_clause': {
                                    'AND': [{
                                        'pred_text': 'has_name',
                                        'obj_text': [0,[28,28]]
                                    }]
                                }
                            }
                        },
                        'relative_direction': 'NEAR'
                    }
                }
            },
            'text_span': 'cube next to the hole' (**What should happen here?**)
        }
    }]
}
</pre>

#### Replacement Follow Up Chat History and LF ###

"User: destory the cube Agent: I'm not sure about something. I think you wanted me to destroy a cube, is that right? User: I mean destroy the blue cube"
<pre>
{
    'dialogue_type': 'HUMAN_GIVE_COMMAND',
    'event_sequence': [{
        'action_type': 'DESTROY',
        'reference_object': {
            'filters': {
                'where_clause': {
                    'AND': [
                        {
                            'pred_text': 'has_name',
                            'obj_text': [0,[28,28]]
                        },
                        {
                            'pred_text': 'has_colour',
                            'obj_text': [0,[27,27]]
                        }
                    ]
                }
            },
            'text_span': 'blue cube' (**What should happen here?**)
        }
    }]
}
</pre>
