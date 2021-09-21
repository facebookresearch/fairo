"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
Q_AGENT = {
    "text": "Who is this command directed towards ?",
    "key": "dialogue_target",
    "tooltip": "e.g. for 'you 4 are in team 1' select 'Specific assistant(s)'",
    "add_radio_other": False,
    "radio": [
        {"text": "A single agent",
         "key": "AGENT",
         "tooltip": "e.g. 'make a cube'"
         },
        {
            "text": "The entire group of assistants or a team",
            "key": "SWARM",
            "tooltip": "e.g. 'everyone form a line'"
        },
        {"text": "Specific assistant(s) or team",
         "key": "f1",
         "tooltip": "e.g. 'you 4 make a tower' or 'alpha and beta go to the house'",
         "next": [
                    {
                        "text": "If a specific number of agents is mentioned, select the words",
                        "key": "filters.selector.return_quantity",
                        "tooltip": "e.g. in 'you 4 build a tower' select '4'",
                        "span": True,
                        "optional": True
                    },
                    {
                        "text": "If agent names are mentioned, select each name individually",
                        "key": "filters.where_clause",
                        "tooltip": "e.g. in 'alpha and beta are in team 1' select 'alpha' and 'beta'",
                        "span": True,
                        "optional": True,
                    },
                     {
                         "text": "If the name of a team is mentioned, select words",
                         "key": "filters.team.where_clause",
                         "tooltip": "e.g. in 'team 1 build a tower' select 'team 1'",
                         "span": True,
                         "optional": True
                     }
                ]
         }
    ]
}


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
                                Q_AGENT,
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
                                                Q_AGENT,
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
                                                Q_AGENT,
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
        # Look or turn or point
        {
            "text": "Turn, look or point",
            "key": "TURN_CHECK",
            "tooltip": "Turn the bot's head, arms or other body parts",
            "next": [
                Q_AGENT,
                {
                    "text": "Which body part is being asked to turn, rotate or move",
                    "key": "TURN_CHECK",
                    "add_radio_other": False,
                    "radio": [
                        # LOOK
                        {
                            "text": "The bot is expected to turn a part of its head, face, camera or eyes",
                            "key": "LOOK",
                            "tooltip": "e.g. 'look at me', 'look left' etc",
                            "next": [
                                {
                                    "text": "Select words specifying where/ how the bot should look or turn its head",
                                    "key": "facing",
                                    "span": True,
                                    "tooltip": "e.g. in 'look at me' select 'me'",
                                }
                            ],
                        },
                        # POINT
                        {
                            "text": "The bot is expected to move its arm to point at something or somewhere",
                            "key": "POINT",
                            "tooltip": "e.g. 'point at the sheep', 'point at me' etc",
                            "next": [
                                {
                                    "text": "Select words where the bot should point",
                                    "key": "facing",
                                    "span": True,
                                    "tooltip": "e.g. in 'point here' select 'here'",
                                }
                            ],
                        },
                        # TURN
                        {
                            "text": "The bot is being asked to turn other body parts.",
                            "key": "TURN",
                            "tooltip": "e.g. 'turn around', 'turn to my left' etc",
                            "next": [
                                {
                                    "text": "Select words specifying where or how the bot should turn",
                                    "key": "facing",
                                    "span": True,
                                    "tooltip": "e.g. in 'turn all the way around' select 'all the way around'",
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        # MOVE
        {
            "text": "Physical movement of some kind",
            "key": "MOVE",
            "tooltip": "The assistant is being asked to move",
            "next": [
                {
                    "text": "Tell us more about the movement type.",
                    "key": "MOVE",
                    "add_radio_other": False,
                    "radio": [
                        # MOVE
                        {
                            "text": "The assistant is being asked to move somewhere.",
                            "key": "yes",
                            "tooltip": "A walk or move to a destination.",
                            "next": [
                                Q_AGENT,
                                {
                                    "text": "Select words specifying the location to which the agent should move",
                                    "key": "location",
                                    "span": True,
                                    "tooltip": "e.g. in 'go to the sheep' select 'to the sheep'",
                                }
                            ],
                        },
                        # DANCE
                        {
                            "text": "This is a movement of some kind where the path or step-sequence is more important than the destination",
                            "key": "no",
                            "tooltip": "Dance or movement where the path is more important than the destination, e.g. go around the cube 4 times",
                            "next": [
                                Q_AGENT,
                                {
                                    "text": "Select words specifying the kind of movement",
                                    "key": "dance_type_span",
                                    "optional": True,
                                    "span": True,
                                    "tooltip": "e.g. in 'jump around the cube' select 'jump'",
                                },
                                {
                                    "text": "Select words specifying where the dance needs to happen",
                                    "key": "location",
                                    "optional": True,
                                    "span": True,
                                    "tooltip": "e.g. in 'dance in front of the cube' select 'in front of the cube'",
                                },
                            ],
                        },
                    ],
                }
            ],
        },
        # SPAWN
        {
            "text": "Spawn something (place an animal or creature in the game world)",
            "key": "SPAWN",
            "tooltip": "for example 'spawn a pig'.",
            "next": [
                Q_AGENT,
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
                Q_AGENT,
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
                Q_AGENT,
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
                Q_AGENT,
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
                    "text": "Tell us more about the object that is being tagged or assigned",
                    "key": "dialogue_target",
                    "tooltip": "e.g. in 'this is bright' select 'this'",
                    "add_radio_other": False,
                    "radio" : [
                        {
                            "text": "The object is a single assistant",
                            "key": "AGENT",
                            "tooltip": "e.g. 'you are in team 1' or 'i will call you alpha'"
                        },
                        {
                            "text": "The object is the entire group of assistants",
                            "key": "SWARM",
                            "tooltip": "e.g. 'you all are team X' or 'i will call you all alpha"
                        },
                        {
                            "text": "The thing being tagged is a specific assistant or multiple assistants",
                            "key": "f1",
                            "tooltip": "e.g. 'you 4 are in team apple' or 'alpha and beta are in team 2",
                            "next": [
                                 {
                                     "text": "If a specific number of agents is mentioned, select the words",
                                     "key": "filters.selector.return_quantity",
                                     "tooltip": "e.g. in 'you 4 build a tower' select '4'",
                                     "span": True,
                                     "optional": True
                                 },
                                 {
                                     "text": "If agent names are mentioned, select each name individually",
                                     "key": "filters.where_clause",
                                     "tooltip": "e.g. in 'alpha and beta are in team 1' select the words:'alpha' and 'beta'",
                                     "span": True,
                                     "optional": True,
                                 }
                            ]
                         },
                        {
                            "text": "Some other object that is not the agent(s), is being tagged",
                            "key": "f2",
                            "next" : [
                                {
                                    "text": "Select words specifying object being tagged or associated",
                                    "key": "filters",
                                    "span": True,
                                    "tooltip": "e.g. in 'this is bright' select 'this'",
                                }
                            ]
                        }
                    ]
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
        {  # NOTE: Add a question for which object is being asked about and another question
            # for what is being asked about. (value of * in Select *)
            # will we ever need the value of * in human_give_command dialogue type or just for
            # GET_MEMORY
            "text": "Answer a question",
            "key": "ANSWER",
            "tooltip": "e.g. 'How many trees are there?' or 'Tell me how deep that tunnel goes'",
        },
        # STOP
        {
            "text": "Stop an action",
            "key": "STOP",
            "tooltip": "Stop or pause something",
            "next": [
                Q_AGENT,
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
                Q_AGENT,
                {
                    "text": "Select words specifying which action needs to be resumed",
                    "key": "target_action_type",
                    "optional": True,
                    "span": True,
                    "tooltip": "e.g. in 'continue walking' select 'walking'",
                }
            ],
        },
        # GET
        {
            "text": "Get, bring or give something",
            "key": "GET",
            "tooltip": "The agent is asked to get, give or bring something.",
            "next": [
                Q_AGENT,
                {
                    "text": "Select words specifying what the agent should get",
                    "key": "reference_object",
                    "span": True,
                    "tooltip": "e.g. in 'get me the dandelion' select 'dandelion'",
                },
                {
                    "text": "Where or who should the object be brought to? (Select an option only if receiver is mentioned)",
                    "key": "receiver",
                    "add_radio_other": False,
                    "optional": True,
                    "tooltip": "e.g. 'bring the box to me', 'get the book here'",
                    "radio": [
                        {
                            "text": "The object needs to be brough to a location mentioned in text",
                            "key": "receiver_loc",
                            "tooltip": "e.g. in 'get the cup here' select 'here'",
                            "next": [
                                {
                                    "text": "Select all words specifying the location where the object should be brought to",
                                    "key": "location",
                                    "span": True,
                                    "tooltip": "e.g. in 'get the dandelion over here' select 'over here'",
                                }
                            ],
                        },
                        {
                            "text": "The object needs to be brought to a person or object",
                            "key": "receiver_ref",
                            "tooltip": "e.g. in 'get the cup to me' select 'me'",
                            "next": [
                                {
                                    "text": "Select all words specifying the reference object to which the object should be brought to",
                                    "key": "reference_object",
                                    "span": True,
                                    "tooltip": "e.g. in 'bring the pen to the chair', select 'chair'",
                                }
                            ],
                        },
                    ],
                },
            ],
        },
        # UNDO
        {
            "text": "Undo or revert an action",
            "key": "UNDO",
            "tooltip": "Undo a previously completed action",
            "next": [
                Q_AGENT,
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
                Q_AGENT,
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
                }
            ],
        },
        {
            "text": "Repeatedly, once for every object or for all objects",
            "key": "repeat_all",
            "tooltip": "e.g. 'Destroy every red block', or 'Build a shed in front of each house'",
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
