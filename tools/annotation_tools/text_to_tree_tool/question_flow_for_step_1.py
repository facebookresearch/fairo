"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
Q_ACTION = {
    "text": 'What action is being requested? If multiple separate actions are being requested (e.g. "do X and then do Y"), select "Multiple separate actions"',
    "key": "action_type",
    "tooltip": "e.g. in 'Make few copies of the cube' it is : 'Build, make a copy or complete something'",
    "add_radio_other": False,
    "radio": [
        # BUILD
        {
            "text": "Build, make a copy or complete something",
            "key": "BUILD",
            "tooltip": "The sentence requests construction or making copies of some object",
            "next": [
                {
                    "text": "Is this exact copy or duplicate of something that already exists?",
                    "key": "COPY",
                    "add_radio_other": False,
                    "radio": [
                        # COPY
                        {
                            "text": "Yes",
                            "key": "yes",
                            "next": [
                                {
                                    "text": "Select words specifying the thing that needs to be copied",
                                    "key": "reference_object",
                                    "tooltip": "e.g. in 'Make 5 copies of the white sheep and put them behind the house' select 'the white sheep'",
                                    "optional": True,
                                    "span": True,
                                },
                                {
                                    "text": "Select words specifying where the copy should be made",
                                    "key": "location",
                                    "tooltip": "e.g. in 'Make 5 copies of the white sheep and put them behind the house' select 'behind the house'",
                                    "span": True,
                                    "optional": True,
                                },
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
                                        # Build
                                        {
                                            "text": "Build a specific object or objects from scratch",
                                            "key": "BUILD",
                                            "next": [
                                                {
                                                    "text": "Select words specifying what needs to be built",
                                                    "key": "schematic",
                                                    "tooltip": "e.g. in 'construct two big wooden houses in front of the tower' select 'big wooden houses'",
                                                    "span": True,
                                                },
                                                {
                                                    "text": "Select words specifying where the thing should be built",
                                                    "key": "location",
                                                    "tooltip": "e.g. in 'construct two big wooden houses in front of the tower' select 'in front of the tower'",
                                                    "span": True,
                                                    "optional": True,
                                                },
                                            ],
                                        },
                                        # Freebuild
                                        {
                                            "text": "Help complete or finish already existing object(s)",
                                            "key": "FREEBUILD",
                                            "next": [
                                                {
                                                    "text": "Select words specifying what needs to be completed",
                                                    "key": "reference_object",
                                                    "tooltip": "e.g. in 'complete that for me please' select 'that'",
                                                    "span": True,
                                                }
                                            ],
                                        },
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        },  # Build , copy, freebuild finishes
        # MOVE
        {
            "text": "Move or walk somewhere",
            "key": "MOVE",
            "tooltip": "The assistant is being asked to move",
            "next": [
                {
                    "text": "Select words specifying the location to which the agent should move",
                    "key": "location",
                    "span": True,
                    "tooltip": "e.g. in 'go to the sheep' select 'to the sheep'",
                }
            ],
        },  # MOVE finishes
        # SPAWN
        {
            "text": "Spawn something (place an animal or creature in the game world)",
            "key": "SPAWN",
            "tooltip": "for example 'spawn a pig'.",
            "next": [
                {
                    "text": "Select words specifying what needs to be spawned",
                    "key": "reference_object",
                    "span": True,
                    "tooltip": "e.g. in 'spawn a pig' select 'a pig' or 'pig'",
                },
                {
                    "text": "Select words specifying where to spawn",
                    "key": "location",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'spawn a pig behind the house' select 'behind the house'",
                },
            ],
        },
        # DESTROY
        {
            "text": "Destroy, remove, or kill something",
            "key": "DESTROY",
            "tooltip": "Something needs to be destroyed.",
            "next": [
                {
                    "text": "Select words specifying what needs to be destroyed",
                    "key": "reference_object",
                    "span": True,
                    "tooltip": "e.g. in 'destroy the red cube over there' select 'red cube over there'",
                }
            ],
        },
        # DIG
        {
            "text": "Dig",
            "key": "DIG",
            "tooltip": "Digging of some kind needs to be done",
            "next": [
                {
                    "text": "Select words specifying what needs to be dug",
                    "key": "schematic",
                    "optional": True,
                    "tooltip": "e.g. in 'dig a big circular hole over there' select 'big circular hole'",
                    "span": True,
                },
                {
                    "text": "Select words specifying where the thing will be dug",
                    "key": "location",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'dig a big hole over there' select 'over there'",
                },
            ],
        },
        # FILL
        {
            "text": "Fill something",
            "key": "FILL",
            "tooltip": "Fill or cover an object/area with something",
            "next": [
                {
                    "text": "Select words specifying what needs to be filled",
                    "key": "reference_object",
                    "span": True,
                    "tooltip": "e.g. in 'fill the mine with diamonds' select 'mine'",
                },
                {
                    "text": "Select words specifying what material is used for filling",
                    "key": "has_block_type",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'fill the mine with diamonds' select 'diamonds'",
                },
            ],
        },
        # Tag
        {
            "text": "Assign a description, name, or tag to an object",
            "key": "TAG",
            "tooltip": "e.g. 'That thing is fluffy' or 'The blue building is my house'",
            "next": [
                {
                    "text": "Select words specifying the object that is being tagged",
                    "key": "filters",
                    "span": True,
                    "tooltip": "e.g. in 'this is bright' select 'this'",
                },
                {
                    "text": "Select words specifying the description or tag being assigned",
                    "key": "tag_val",
                    "span": True,
                    "tooltip": "e.g. in 'this is bright' select 'bright'",
                },
            ],
        },
        # Answer
        {
            "text": "Answer a question",
            "key": "ANSWER",
            "tooltip": "e.g. 'How many trees are there?' or 'Tell me how deep that tunnel goes'",
        },
        # Dance
        {
            "text": "A movement where the path or step-sequence is more important than the destination",
            "key": "DANCE",
            "tooltip": "Dance or movement where the path is more important than the destination, e.g. go around the cube 4 times",
            "next": [
                {
                    "text": "Select words specifying where the dance needs to happen",
                    "key": "location",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'dance in front of the cube' select 'in front of the cube'",
                }
            ],
        },
        # STOP
        {
            "text": "Stop an action",
            "key": "STOP",
            "tooltip": "Stop or pause something",
            "next": [
                {
                    "text": "Select words specifying which action needs to be stopped",
                    "key": "target_action_type",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'stop digging' select 'digging'",
                }
            ],
        },
        # RESUME
        {
            "text": "Resume an action",
            "key": "RESUME",
            "tooltip": "Resume or continue something",
            "next": [
                {
                    "text": "Select words specifying which action needs to be resumed",
                    "key": "target_action_type",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'continue walking' select 'walking'",
                }
            ],
        },
        # UNDO
        {
            "text": "Undo or revert an action",
            "key": "UNDO",
            "tooltip": "Undo a previously completed action",
            "next": [
                {
                    "text": "Select words specifying which action needs to be reverted",
                    "key": "target_action_type",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'undo what you built' select 'what you built'",
                }
            ],
        },
        # MULTIPLE ACTIONS
        {
            "text": "Multiple separate actions",
            "key": "COMPOSITE_ACTION",
            "tooltip": "e.g. 'Build a cube and then run around'. Do not select this for a single repeated action, e.g. 'Build 5 cubes'",
        },
        # OTHER ACTION NOT LISTED
        {
            "text": "Another action not listed here",
            "key": "OtherAction",
            "tooltip": "In case the given sentence is a command, but not one of the command types listed above, please click this",
            "next": [
                {
                    "text": "What object (if any) is the target of this action?",
                    "key": "reference_object",
                    "span": True,
                    "optional": True,
                    "tooltip": "e.g. in 'Sharpen the axe behind me', select 'axe'",
                },
                {
                    "text": "Where should the action take place?",
                    "key": "location",
                    "span": True,
                    "optional": True,
                    "tooltip": "e.g. in 'Sharpen the axe behind me', select 'behind me'",
                },
            ],
        },
        # NOT ACTION
        {
            "text": "This sentence is not a command or request to do something",
            "key": "NOOP",
            "tooltip": "In case the given sentence is not a command or request to do something, please click this",
        },
    ],
}

REPEAT_DIR = [
    {"text": "Not specified", "key": None, "tooltip": "The direction isn't specified"},
    {
        "text": "Forward",
        "key": "FRONT",
        "tooltip": "Repetition towards the front / forward direction",
    },
    {"text": "Backward", "key": "BACK", "tooltip": "Repetition towards the back"},
    {"text": "Left", "key": "LEFT", "tooltip": "Repetition to the left"},
    {"text": "Right", "key": "RIGHT", "tooltip": "Repetition to the left"},
    {"text": "Up", "key": "UP", "tooltip": "Repetition upwards"},
    {"text": "Down", "key": "DOWN", "tooltip": "Repetition downward"},
    {"text": "Around", "key": "AROUND", "tooltip": "Repetition around"},
]


Q_ACTION_LOOP = {
    "text": "How many times should this action be performed?",
    "key": "loop",
    "tooltip": "Does the above action or any part of it need to be repeated ?",
    "add_radio_other": False,
    "radio": [
        {
            "text": "Just once, or not specified",
            "key": None,
            "tooltip": "No repeats needed, it needs to be done exactly once.",
        },
        {
            "text": "Repeatedly, a specific number of times",
            "key": "ntimes",
            "tooltip": "The action needs to be repeated a fixed number of times",
            "next": [
                {
                    "text": "How many times? Select all words",
                    "span": True,
                    "key": "repeat_for",
                    "tooltip": "e.g. in 'go around the cube twice' select 'twice'",
                },
                {
                    "text": "In which direction should the action be repeated?",
                    "key": "repeat_dir",
                    "radio": REPEAT_DIR,
                    "tooltip": "e.g. in 'go around the cube twice' select 'Around'",
                },
            ],
        },
        {
            "text": "Repeatedly, once for every object or for all objects",
            "key": "repeat_all",
            "tooltip": "e.g. 'Destroy every red block', or 'Build a shed in front of each house'",
            "next": [
                {
                    "text": "In which direction should the action be repeated?",
                    "key": "repeat_dir",
                    "radio": REPEAT_DIR,
                    "tooltip": "e.g. in 'stack 5 blocks' select 'Up' since stacking is done upwards",
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
            "tooltip": "e.g. 'Dig until you hit bedrock', 'Keep walking until you reach water'",
            "next": [
                {
                    "text": "Until the assistant reaches a specific object(s) / area",
                    "key": "adjacent_to_block_type",
                    "optional": True,
                    "tooltip": "e.g. in 'Dig until you hit bedrock' select 'bedrock'",
                    "span": True,
                },
                {
                    "text": "Until some other condition is met",
                    "key": "condition_span",
                    "optional": True,
                    "tooltip": "e.g. in 'Keep building until it is sundown' select 'it is sundown'",
                    "span": True,
                },
            ],
        },
    ],
}
