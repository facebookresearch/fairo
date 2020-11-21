"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

LOCATION_RADIO = [
    {"text": "Not specified", "key": None},
    {
        "text": "Represented using word(s) that indicate reference to a location (e.g. 'there', 'here', 'over there' etc)",
        "key": "coref_resolve_check",
        "tooltip": "e.g. 'there', 'here', 'over there' etc",
        "next": [
            {
                "text": "What are the word(s) representing the location?",
                "key": "yes.coref_resolve",
                "span": True,
            }
        ],
    },
    {"text": "Where the speaker is looking (e.g. 'where I am looking')", "key": "SPEAKER_LOOK"},
    {"text": "Where the speaker is standing (e.g. 'by me', 'where I am')", "key": "SpeakerPos"},
    {
        "text": "Where the assistant is standing (e.g. 'by you', 'where you are', 'where you are standing')",
        "key": "AGENT_POS",
    },
    {
        "text": "Somewhere relative to another object(s) / area(s)",
        "key": "REFERENCE_OBJECT",
        "next": [
            {
                "text": "In terms of number of steps, how many ?",
                "key": "steps",
                "span": True,
                "optional": True,
            },
            {
                "text": "Where in relation to the other object(s)?",
                "key": "relative_direction",
                "radio": [
                    {"text": "Left", "key": "LEFT"},
                    {"text": "Right", "key": "RIGHT"},
                    {"text": "Above", "key": "UP"},
                    {"text": "Below", "key": "DOWN"},
                    {"text": "In front", "key": "FRONT"},
                    {"text": "Behind", "key": "BACK"},
                    {"text": "Away from", "key": "AWAY"},
                    {"text": "Inside", "key": "INSIDE"},
                    {"text": "Outside", "key": "OUTSIDE"},
                    {"text": "Nearby or close to", "key": "NEAR"},
                    {"text": "Around", "key": "AROUND"},
                    {"text": "Exactly at", "key": "EXACT"},
                ],
            },
            {
                "text": "Are there words or pronouns that represent the relative object (e.g. 'this', 'that', 'these', 'those', 'it' etc)?",
                "key": "coref_resolve_check",
                "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                "radio": [
                    {
                        "text": "Yes",
                        "key": "yes",
                        "next": [
                            {
                                "text": "What is the word / pronoun representing the object?",
                                "key": "coref_resolve",
                                "span": True,
                            }
                        ],
                    },
                    {
                        "text": "No",
                        "key": "no",
                        "next": [
                            {
                                "text": "What is the name of the relative object(s) or area?",
                                "key": "has_name",
                                "span": True,
                            }
                        ],
                    },
                ],
            },
        ],
    },
]

"""
{
    "text": "Can this object(s) be represented using a pronoun? If so, what is it",
    "key": "reference_object.coref_resolve",
    "span": True,
    "optional": True,
},
"""
REF_OBJECT_OPTIONALS = [
    {
        "text": "What is the building material?",
        "key": "reference_object.has_block_type",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the color?",
        "key": "reference_object.has_colour",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the size?",
        "key": "reference_object.has_size",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the width?",
        "key": "reference_object.has_width",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the height?",
        "key": "reference_object.has_height",
        "span": True,
        "optional": True,
    },
    {
        "text": "What is the depth?",
        "key": "reference_object.has_depth",
        "span": True,
        "optional": True,
    },
]


Q_ACTION = {
    "text": 'What action is being instructed? If multiple separate actions are being instructed (e.g. "do X and then do Y"), select "Multiple separate actions"',
    "key": "action_type",
    "add_radio_other": False,
    "radio": [
        # BUILD
        {
            "text": "Build, copy or complete something",
            "key": "BUILD",
            "next": [
                {
                    "text": "Is this an exact copy or duplicate of an existing object?",
                    "key": "COPY",
                    "add_radio_other": False,
                    "radio": [
                        # COPY
                        {
                            "text": "Yes",
                            "key": "yes",
                            "next": [
                                {
                                    "text": "Are there words or pronouns that indicate reference to the object to be copied (e.g. 'this', 'that', 'these', 'those', 'it' etc)",
                                    "key": "reference_object.coref_resolve_check",
                                    "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                                    "add_radio_other": False,
                                    "radio": [
                                        {
                                            "text": "Yes",
                                            "key": "yes",
                                            "next": [
                                                {
                                                    "text": "What is the word / pronoun representing the object?",
                                                    "key": "coref_resolve",
                                                    "span": True,
                                                }
                                            ],
                                        },
                                        {
                                            "text": "No",
                                            "key": "no",
                                            "next": [
                                                {
                                                    "text": "What is the name of the object that should be copied?",
                                                    "key": "has_name",
                                                    "span": True,
                                                }
                                            ],
                                        },
                                    ],
                                },
                                *REF_OBJECT_OPTIONALS,
                                # {
                                #     "text": "What is the location of the object to be copied?",
                                #     "key": "location",
                                #     "radio": LOCATION_RADIO,
                                # },
                            ],
                        },
                        # BUILD
                        {
                            "text": "No",
                            "key": "no",
                            "next": [
                                {
                                    "text": "Is the assistant being asked to...",
                                    "key": "FREEBUILD",
                                    "add_radio_other": False,
                                    "radio": [
                                        {
                                            "text": "Build a fresh complete, specific object",
                                            "key": "BUILD",
                                            "next": [
                                                {
                                                    "text": "What is the name of the thing to be built ?",
                                                    "key": "schematic.has_name",
                                                    "span": True,
                                                },
                                                {
                                                    "text": "What is the building material (what should it be built out of)?",
                                                    "key": "schematic.has_block_type",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the size?",
                                                    "key": "schematic.has_size",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the width?",
                                                    "key": "schematic.has_width",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the colour ?",
                                                    "key": "schematic.has_colour",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the height?",
                                                    "key": "schematic.has_height",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the depth?",
                                                    "key": "schematic.has_depth",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                                {
                                                    "text": "What is the thickness?",
                                                    "key": "schematic.has_thickness",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                            ],
                                        },
                                        {
                                            "text": "Help complete or finish an already existing object",
                                            "key": "FREEBUILD",
                                            "next": [
                                                {
                                                    "text": "Are there word(s) / pronouns that indicate reference to the object to be completed (e.g. 'this', 'that', 'these', 'those', 'it' etc)?",
                                                    "key": "reference_object.coref_resolve_check",
                                                    "add_radio_other": False,
                                                    "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                                                    "radio": [
                                                        {
                                                            "text": "Yes",
                                                            "key": "yes",
                                                            "next": [
                                                                {
                                                                    "text": "What is the word / pronoun representing the object?",
                                                                    "key": "coref_resolve",
                                                                    "span": True,
                                                                }
                                                            ],
                                                        },
                                                        {
                                                            "text": "No",
                                                            "key": "no",
                                                            "next": [
                                                                {
                                                                    "text": "What is the name of the object that should be completed?",
                                                                    "key": "has_name",
                                                                    "span": True,
                                                                }
                                                            ],
                                                        },
                                                    ],
                                                },
                                                *REF_OBJECT_OPTIONALS,
                                            ],
                                        },
                                    ],
                                }
                            ],
                        },
                    ],
                },
                {
                    "text": "Where should the construction / copying / completion happen?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # MOVE
        {
            "text": "Move or walk somewhere",
            "key": "MOVE",
            "next": [
                {
                    "text": "Where should the assistant move to?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                }
            ],
        },
        # SPAWN
        {
            "text": "Spawn something",
            "key": "SPAWN",
            "next": [
                {
                    "text": "What is the name of the object that should be spawned?",
                    "key": "reference_object.has_name",
                    "span": True,
                }
            ],
        },
        # DESTROY
        {
            "text": "Destroy, remove, or kill something",
            "key": "DESTROY",
            "next": [
                {
                    "text": "Are there word(s) / pronouns that indicate reference to the object to be destroyed (e.g. 'this', 'that', 'these', 'those', 'it' etc)?",
                    "key": "reference_object.coref_resolve_check",
                    "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                    "radio": [
                        {
                            "text": "Yes",
                            "key": "yes",
                            "next": [
                                {
                                    "text": "What is the word / pronoun representing the object?",
                                    "key": "coref_resolve",
                                    "span": True,
                                }
                            ],
                        },
                        {
                            "text": "No",
                            "key": "no",
                            "next": [
                                {
                                    "text": "What is the name of the object that should be destroyed?",
                                    "key": "has_name",
                                    "span": True,
                                }
                            ],
                        },
                    ],
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "What is the location of the object to be removed?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # DIG
        {
            "text": "Dig",
            "key": "DIG",
            "next": [
                {
                    "text": "Where should the digging happen ?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
                {
                    "text": "What is the size of the thing to be dug?",
                    "key": "has_size",
                    "span": True,
                    "optional": True,
                },
                {
                    "text": "What is the width of the thing to be dug?",
                    "key": "has_width",
                    "span": True,
                    "optional": True,
                },
                {
                    "text": "What is the height of the thing to be dug?",
                    "key": "has_length",
                    "span": True,
                    "optional": True,
                },
                {
                    "text": "What is the depth of the thing to be dug?",
                    "key": "has_depth",
                    "span": True,
                    "optional": True,
                },
            ],
        },
        # FILL
        {
            "text": "Fill something",
            "key": "FILL",
            "next": [
                {
                    "text": "Where is the thing that should be filled ?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
                {
                    "text": "What should the thing be filled with?",
                    "key": "has_block_type",
                    "span": True,
                    "optional": True,
                },
                *REF_OBJECT_OPTIONALS,
            ],
        },
        # TAG
        {
            "text": "Assign a description, name, or tag to an object",
            "key": "TAG",
            "tooltip": "e.g. 'That thing is fluffy' or 'The blue building is my house'",
            "next": [
                {
                    "text": "What is the description, name, or tag being assigned?",
                    "key": "tag",
                    "span": True,
                },
                {
                    "text": "What object is being assigned a description, name, or tag?",
                    "key": "reference_object",
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "What is the location of the object to be described, named, or tagged?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # STOP
        {
            "text": "Stop an action",
            "key": "STOP",
            "next": [
                {
                    "text": "Is this a command to stop a particular action?",
                    "key": "target_action_type",
                    "radio": [
                        {"text": "Building or copying", "key": "BUILD"},
                        {"text": "Moving", "key": "MOVE"},
                        {"text": "Destroying", "key": "DESTROY"},
                        {"text": "Digging", "key": "DIG"},
                        {"text": "Filling", "key": "FILL"},
                    ],
                }
            ],
        },
        # RESUME
        {
            "text": "Resume an action",
            "key": "RESUME",
            "next": [
                {
                    "text": "Is this a command to resume a particular action?",
                    "key": "target_action_type",
                    "radio": [
                        {"text": "Building or copying", "key": "BUILD"},
                        {"text": "Moving", "key": "MOVE"},
                        {"text": "Destroying", "key": "DESTROY"},
                        {"text": "Digging", "key": "DIG"},
                        {"text": "Filling", "key": "FILL"},
                    ],
                }
            ],
        },
        # UNDO
        {
            "text": "Undo or revert an action",
            "key": "UNDO",
            "next": [
                {
                    "text": "Is this a command to undo a particular action?",
                    "key": "target_action_type",
                    "radio": [
                        {"text": "Building", "key": "BUILD"},
                        {"text": "Destroying", "key": "DESTROY"},
                        {"text": "Digging", "key": "DIG"},
                        {"text": "Filling", "key": "FILL"},
                    ],
                }
            ],
        },
        # ANSWER QUESTION
        {
            "text": "Answer a question",
            "key": "ANSWER",
            "tooltip": "e.g. 'How many trees are there?' or 'Tell me how deep that tunnel goes'",
            "next": [
                {
                    "text": "What is being asked about ?",
                    "key": "filters",
                    "radio": [
                        {
                            "text": "Where the assistant is heading",
                            "key": "type.AGENT.move_target",  # assign TAG, move_target
                        },
                        {
                            "text": "Where the assistant is currently located",
                            "key": "type.AGENT.location",  # assign TAG, location
                        },
                        {
                            "text": "Name of the action the assistant is performing",
                            "key": "type.ACTION.action_name",  # assign TAG, action_name
                        },
                        {
                            "text": "Name of the object that an action is being performed on",
                            "key": "type.ACTION.action_reference_object_name",  # # assign TAG, action_reference_object_name
                            "next": [
                                {
                                    "text": "Which action is being asked about?",
                                    "key": "action_type",
                                    "radio": [
                                        {"text": "Building", "key": "BUILD"},
                                        {"text": "Destroying", "key": "DETSROY"},
                                        {"text": "Digging", "key": "DIG"},
                                        {"text": "Filling", "key": "FILL"},
                                        {"text": "Spawning", "key": "SPAWN"},
                                        {"text": "Moving", "key": "MOVE"},
                                    ],
                                }
                            ],
                        },
                        {
                            "text": "Questions related to a specific object(s) / area(s)",
                            "key": "type.REFERENCE_OBJECT",
                            "next": [
                                {
                                    "text": "Is this a yes/no question ?",
                                    "key": "answer_type",
                                    "radio": [
                                        {"text": "Yes this is a yes/no question", "key": "EXISTS"},
                                        {
                                            "text": "No some specific attribute is being asked about",
                                            "key": "TAG",
                                            "next": [
                                                {
                                                    "text": "What exactly is being asked about",
                                                    "key": "tag_name",
                                                    "radio": [
                                                        {"text": "The name", "key": "has_name"},
                                                        {"text": "The size", "key": "has_size"},
                                                        {
                                                            "text": "The colour",
                                                            "key": "has_colour",
                                                        },
                                                    ],
                                                }
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "text": "Are there words or pronouns that represent the object being talked about(e.g. 'this', 'that', 'these', 'those', 'it' etc)?",
                                    "key": "coref_resolve_check",
                                    "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                                    "radio": [
                                        {
                                            "text": "Yes",
                                            "key": "yes",
                                            "next": [
                                                {
                                                    "text": "What is the word / pronoun representing the object?",
                                                    "key": "coref_resolve",
                                                    "span": True,
                                                }
                                            ],
                                        },
                                        {
                                            "text": "No",
                                            "key": "no",
                                            "next": [
                                                {
                                                    "text": "What is the name of the object(s) being asked about?",
                                                    "key": "has_name",
                                                    "span": True,
                                                }
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "text": "Is the building material being asked about?",
                                    "key": "has_block_type",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the color of the object being asked about?",
                                    "key": "has_colour",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the size of the object being asked about?",
                                    "key": "has_size",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the width of the object being asked about?",
                                    "key": "has_width",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the height of the object being asked about?",
                                    "key": "has_height",
                                    "span": True,
                                    "optional": True,
                                },
                                {
                                    "text": "Is the depth of the object being asked about?",
                                    "key": "has_depth",
                                    "span": True,
                                    "optional": True,
                                },
                            ],
                        },
                    ],
                }
            ],
        },
        # OTHER ACTION NOT LISTED
        {
            "text": "Another action not listed here",
            "key": "OtherAction",
            "tooltip": "The sentence is a command, but not one of the actions listed here",
            "next": [
                {
                    "text": "What object (if any) is the target of this action? e.g. for the sentence 'Sharpen this axe', select the word 'axe'",
                    "key": "reference_object.has_name",
                    "span": True,
                },
                *REF_OBJECT_OPTIONALS,
                {
                    "text": "Where should the action take place?",
                    "key": "location",
                    "radio": LOCATION_RADIO,
                },
            ],
        },
        # NOT ACTION
        {
            "text": "This sentence is not a command or request to do something",
            "key": "NOOP",
            "tooltip": "e.g. 'Yes', 'Hello', or 'What a nice day it is today'",
        },
        # MULTIPLE ACTIONS
        {
            "text": "Multiple separate actions",
            "key": "COMPOSITE_ACTION",
            "tooltip": "e.g. 'Build a cube and then run around'. Do not select this for a single repeated action, e.g. 'Build 5 cubes'",
        },
    ],
}


REPEAT_DIR = [
    {"text": "Not specified", "key": None},
    {"text": "Forward", "key": "FRONT"},
    {"text": "Backward", "key": "BACK"},
    {"text": "Left", "key": "LEFT"},
    {"text": "Right", "key": "RIGHT"},
    {"text": "Up", "key": "UP"},
    {"text": "Down", "key": "DOWN"},
    {"text": "Around", "key": "AROUND"},
]


Q_ACTION_LOOP = {
    "text": "How many times should this action be performed?",
    "key": "loop",
    "radio": [
        {"text": "Just once, or not specified", "key": None},
        {
            "text": "Repeatedly, a specific number of times",
            "key": "ntimes",
            "next": [
                {"text": "How many times?", "span": True, "key": "repeat_for"},
                {
                    "text": "In which direction should the action be repeated?",
                    "key": "repeat_dir",
                    "radio": REPEAT_DIR,
                },
            ],
        },
        {
            "text": "Repeatedly, once for every object or for all objects",
            "key": "repeat_all",
            "tooltip": "e.g. 'Destroy the red blocks', or 'Build a shed in front of each house'",
            "next": [
                {
                    "text": "In which direction should the action be repeated?",
                    "key": "repeat_dir",
                    "radio": REPEAT_DIR,
                }
            ],
        },
        {
            "text": "Repeated forever",
            "key": "forever",
            "tooltip": "e.g. 'Keep building railroad tracks in that direction' or 'Collect diamonds until I tell you to stop'",
        },
        {
            "text": "Repeated until a certain condition is met",
            "key": "repeat_until",
            # "tooltip": "e.g. 'Dig until you hit bedrock', 'Keep walking until you reach water'",
            "next": [
                {
                    "text": "Until the assistant reaches some object(s) /area",
                    "key": "adjacent_to_block_type",
                    "span": True,
                }
            ],
        },
    ],
}
