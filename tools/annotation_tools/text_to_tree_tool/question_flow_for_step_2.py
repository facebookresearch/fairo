"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# 'location', 'move'
# 'reference_object', 'spawn'
# 'reference_object', 'destroy'
# 'schematic', 'dig'
# 'reference_object', fill
# 'reference_object', 'OtherAction'
# 'location', 'OtherAction'
# 'target_action_type', 'stop'
# 'target_action_type', 'resume'
# 'target_action_type', 'undo'

LOCATION_RADIO = [
    {"text": "Not specified", "key": None, "tooltip": "The location information is missing."},
    {
        "text": "The location is represented using an indefinite noun like 'there' or 'over here'",
        "key": "CONTAINS_COREFERENCE",
        "tooltip": "e.g. 'there', 'here', 'over there' etc",
    },
    {
        "text": "Exact numerical coordinates are given",
        "key": "coordinates_check",
        "tooltip": "Exact numeric coordinates are specified.",
        "next": [
            {
                "text": "Click on all words representing the coordinates",
                "key": "yes.coordinates",
                "span": True,
                "tooltip": "e.g. in 'make a box at 4 , 5 , 6' select all: '4 , 5 , 6'",
            }
        ],
    },
    {
        "text": "Where the speaker is looking",
        "key": "SPEAKER_LOOK",
        "tooltip": "e.g. 'where I am looking'",
    },
    {
        "text": "Somewhere relative to where the speaker is looking",
        "key": "SPEAKER_LOOK_REL",
        "tooltip": "e.g. 'in front of where I am looking'",
        "next": [
            {
                "text": "Where (which direction) in relation to where the speaker is looking?",
                "key": "relative_direction",
                "radio": [
                    {"text": "Left", "key": "LEFT"},
                    {"text": "Right", "key": "RIGHT"},
                    {"text": "Above", "key": "UP"},
                    {"text": "Below", "key": "DOWN"},
                    {"text": "In front", "key": "FRONT"},
                    {"text": "Behind", "key": "BACK"},
                    {"text": "Away from", "key": "AWAY"},
                    {"text": "Nearby or close to", "key": "NEAR"},
                    {"text": "Around", "key": "AROUND"},
                    {"text": "Exactly at", "key": "EXACT"},
                ],
            }
        ],
    },
    {
        "text": "Where the speaker is standing",
        "key": "SPEAKER_POS",
        "tooltip": "e.g. 'by me', 'where I am', 'where I am standing'",
    },
    {
        "text": "Somewhere relative to where the speaker is standing",
        "key": "SPEAKER_POS_REL",
        "tooltip": "e.g. 'in front of where I am', 'behind me'",
        "next": [
            {
                "text": "Where (which direction) in relation to where the speaker is standing?",
                "key": "relative_direction",
                "radio": [
                    {"text": "Left", "key": "LEFT"},
                    {"text": "Right", "key": "RIGHT"},
                    {"text": "Above", "key": "UP"},
                    {"text": "Below", "key": "DOWN"},
                    {"text": "In front", "key": "FRONT"},
                    {"text": "Behind", "key": "BACK"},
                    {"text": "Away from", "key": "AWAY"},
                    {"text": "Nearby or close to", "key": "NEAR"},
                    {"text": "Around", "key": "AROUND"},
                    {"text": "Exactly at", "key": "EXACT"},
                ],
            }
        ],
    },
    {
        "text": "Where the assistant is standing",
        "key": "AGENT_POS",
        "tooltip": "e.g. 'by you', 'where you are', 'where you are standing'",
    },
    {
        "text": "Somewhere relative to where the assistant is standing",
        "key": "AGENT_POS_REL",
        "tooltip": "e.g. 'in front of you', 'behind you'",
        "next": [
            {
                "text": "Where (which direction) in relation to where the assistant is standing?",
                "key": "relative_direction",
                "radio": [
                    {"text": "Left", "key": "LEFT"},
                    {"text": "Right", "key": "RIGHT"},
                    {"text": "Above", "key": "UP"},
                    {"text": "Below", "key": "DOWN"},
                    {"text": "In front", "key": "FRONT"},
                    {"text": "Behind", "key": "BACK"},
                    {"text": "Away from", "key": "AWAY"},
                    {"text": "Nearby or close to", "key": "NEAR"},
                    {"text": "Around", "key": "AROUND"},
                    {"text": "Exactly at", "key": "EXACT"},
                ],
            }
        ],
    },
]

LOCATION_REL_OBJECT_QUESTIONS = [
    {
        "text": "Click on all words specifying the object / area relative of which the location is given",
        "key": "reference_object.has_name",
        "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
        "span": True,
    },
    {
        "text": "Are there indefinite nouns or pronouns specifying the relative object?",
        "key": "reference_object.contains_coreference",
        "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
        "add_radio_other": False,
        "radio": [{"text": "Yes", "key": "yes"}, {"text": "No", "key": "no"}],
    },
    # {
    #     "text": "Is the location of the reference object mentioned ? Select all words.",
    #     "key": "reference_object.location",
    #     "span": True,
    #     "optional": True,
    #     "tooltip": "e.g. in 'to the right of the cow behind the house' select 'behind the house'",
    #     # "radio": LOCATION_RADIO,
    # },
]


LOCATION_REL_OBJECT = [
    {
        "text": "Somewhere relative to (or exactly at) another object(s) / area(s)",
        "key": "REFERENCE_OBJECT",
        "next": [
            {
                "text": "Where (which direction) in relation to the other object(s)?",
                "key": "relative_direction",
                "radio": [
                    {
                        "text": "Left or towards the west direction",
                        "key": "LEFT",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area to the left of which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Right or towards the east direction",
                        "key": "RIGHT",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area to the right of which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Above or towards the north direction",
                        "key": "UP",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area above which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Below or towards the south direction",
                        "key": "DOWN",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area below which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "In front",
                        "key": "FRONT",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area in front of which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Behind",
                        "key": "BACK",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area at the back of which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Away from",
                        "key": "AWAY",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area away from which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Inside",
                        "key": "INSIDE",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area inside which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Outside",
                        "key": "OUTSIDE",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area outside which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Between two object(s) / area(s)",
                        "key": "BETWEEN",
                        "next": [
                            {
                                "text": "Click on all words specifying the first object / area relative to which the location is given",
                                "key": "reference_object_1.has_name",
                                "tooltip": "e.g. in 'make 5 copies between the car and the house' select 'car'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the first relative object?",
                                "key": "reference_object_1.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                            {
                                "text": "Click on all words specifying the second object / area relative to which the location is given",
                                "key": "reference_object_2.has_name",
                                "tooltip": "e.g. in 'make 5 copies between the car and the house' select 'house'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the second relative object?",
                                "key": "reference_object_2.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Nearby or close to",
                        "key": "NEAR",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area close to which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Around",
                        "key": "AROUND",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area around which the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                    {
                        "text": "Exactly at",
                        "key": "EXACT",
                        "next": [
                            {
                                "text": "Click on all words specifying the object / area exactly where the location is given",
                                "key": "reference_object.has_name",
                                "tooltip": "e.g. in 'make 5 copies to the left of the cow' select 'cow'",
                                "span": True,
                            },
                            {
                                "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                                "key": "reference_object.contains_coreference",
                                "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                                "add_radio_other": False,
                                "radio": [
                                    {"text": "Yes", "key": "yes"},
                                    {"text": "No", "key": "no"},
                                ],
                            },
                        ],
                    },
                ],
            }
        ],
    }
]

REF_OBJECT_OPTIONALS = [
    {
        "text": "What is the building material?",
        "key": "reference_object.has_block_type",
        "span": True,
        "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'glass'",
    },
    {
        "text": "What is the color?",
        "key": "reference_object.has_colour",
        "span": True,
        "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'blue'",
    },
    {
        "text": "What is the size?",
        "key": "reference_object.has_size",
        "span": True,
        "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'tiny'",
    },
    {
        "text": "What is the width?",
        "key": "reference_object.has_width",
        "span": True,
        "tooltip": "e.g. in 'next to the 5 step wide hole' select '5'",
    },
    {
        "text": "What is the height?",
        "key": "reference_object.has_height",
        "span": True,
        "tooltip": "e.g. in 'next to the tower that is 20 blocks high' select '20'",
    },
    {
        "text": "What is the depth?",
        "key": "reference_object.has_depth",
        "span": True,
        "tooltip": "e.g. in 'fill the 20 block deep hole for me' select '20'",
    },
]


def get_questions(child, action, optional_words=None):
    QUESTION = None

    if child == "schematic":
        if action == "build":
            # add tooltip
            QUESTION = {
                "text": "Click on all properties of the thing to be built mentioned in the highlighted text.",
                "key": "schematic",
                "checkbox": True,
                "tooltip": "e.g. in 'make a blue square' click on 'Name' as well as 'Colour' since both are specified in 'blue square'",
                "add_radio_other": False,
                "radio": [
                    {
                        "text": "Name",
                        "key": "name_check",
                        "tooltip": "Select this if the name of the thing to be built is mentioned",
                        "next": [
                            {
                                "text": "Select all words that indicate the name of the thing to be built",
                                "key": "has_name",
                                "tooltip": "e.g. in 'Build a big green wooden house there' select 'house'",
                                "span": True,
                            }
                        ],
                    },
                    {
                        "text": "Abstract/non-numeric size (e.g. 'big', 'small', etc.)",
                        "key": "size_check",
                        "tooltip": "Select this if the size of the thing to be built is specified",
                        "next": [
                            {
                                "text": "Select all words that represent the size",
                                "key": "has_size",
                                "span": True,
                                "tooltip": "e.g. in 'Build a big green wooden house there' select 'big'",
                            }
                        ],
                    },
                    {
                        "text": "Colour",
                        "key": "colour_check",
                        "tooltip": "Select this if the colour of what needs to be built is specified",
                        "next": [
                            {
                                "text": "Select all words that represent the colour.",
                                "key": "has_colour",
                                "span": True,
                                "tooltip": "e.g. in 'Build a big green wooden house there' select 'green'",
                            }
                        ],
                    },
                    {
                        "text": "The building material",
                        "key": "block_type_check",
                        "tooltip": "Select this if the building material is mentioned",
                        "next": [
                            {
                                "text": "What should it be built out of? Select all words.",
                                "key": "has_block_type",
                                "tooltip": "e.g. in 'Build a big green wooden house there' select 'wooden'",
                                "span": True,
                            }
                        ],
                    },
                    {
                        "text": "Height",
                        "key": "height_check",
                        "tooltip": "Select this if the height is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for height.",
                                "key": "has_height",
                                "span": True,
                                "tooltip": "e.g. in 'make a 5 block tall tower here' select '5'",
                            }
                        ],
                    },
                    {
                        "text": "Width",
                        "key": "width_check",
                        "tooltip": "Select this if the width is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for width",
                                "key": "has_width",
                                "span": True,
                                "tooltip": "e.g. in 'make a 4 blocks wide square there' select '4'",
                            }
                        ],
                    },
                    {
                        "text": "Length",
                        "key": "length_check",
                        "tooltip": "Select this if the length is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for length",
                                "key": "has_length",
                                "span": True,
                                "tooltip": "e.g. in 'make a 4 blocks long square there' select '4'",
                            }
                        ],
                    },
                    {
                        "text": "Thickness",
                        "key": "thickness_check",
                        "tooltip": "Select this if the thickness is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for thickness",
                                "key": "has_thickness",
                                "span": True,
                                "tooltip": "e.g. in 'make a hollow rectangle of thickness 3' select '3'",
                            }
                        ],
                    },
                    {
                        "text": "Some other property not mentioned above",
                        "key": "tag_check",
                        "tooltip": "Select this if any propoerty not explicitly mentioned above is given",
                        "next": [
                            {
                                "text": "Select all words for this property",
                                "key": "has_tag",
                                "span": True,
                                "tooltip": "e.g. in 'make a bright cabin' select 'bright'",
                            }
                        ],
                    },
                ],
            }
        elif action == "dig":
            # add tooltip
            QUESTION = {
                "text": "Click on all properties of the thing to be dug mentioned in the highlighted text.",
                "key": "schematic",
                "checkbox": True,
                "tooltip": "e.g. in 'dig a 10 x 10 pool' click on 'Name' as well as 'length' and 'width' since all are specified in '10 x 10 pool'",
                "add_radio_other": False,
                "radio": [
                    {
                        "text": "Name",
                        "key": "name_check",
                        "tooltip": "Select this if the name of the thing to be dug is mentioned",
                        "next": [
                            {
                                "text": "Select all words that indicate the name of the thing to be dug",
                                "key": "has_name",
                                "tooltip": "e.g. in 'dig a 10 x 10 pool there' select 'pool'",
                                "span": True,
                            }
                        ],
                    },
                    {
                        "text": "Length",
                        "key": "length_check",
                        "tooltip": "Select this if the length is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for length.",
                                "key": "has_length",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 5 feet by 5 feet hole here' select '5'",
                            }
                        ],
                    },
                    {
                        "text": "Width",
                        "key": "width_check",
                        "tooltip": "Select this if the width is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for width",
                                "key": "has_width",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 2 by 3 hole there' select '3'",
                            }
                        ],
                    },
                    {
                        "text": "Depth",
                        "key": "depth_check",
                        "tooltip": "Select this if the depth is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for depth",
                                "key": "has_depth",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 1 x 2 x 3 pool' select '3'",
                            }
                        ],
                    },
                    {
                        "text": "Abstract/non-numeric size (e.g. 'big', 'small', etc.)",
                        "key": "size_check",
                        "tooltip": "Select this if the size of the thing to be dug is specified without number words",
                        "next": [
                            {
                                "text": "Select all words that describe the abstract size",
                                "key": "has_size",
                                "span": True,
                                "tooltip": "e.g. in 'dig a big hole' select 'big'",
                            }
                        ],
                    },
                    {
                        "text": "Some other property not mentioned above",
                        "key": "tag_check",
                        "tooltip": "Select this if any propoerty not explicitly mentioned above is given",
                        "next": [
                            {
                                "text": "Select all words for this property",
                                "key": "has_tag",
                                "span": True,
                                "tooltip": "e.g. in 'make a bright cabin' select 'bright'",
                            }
                        ],
                    },
                ],
            }
    elif child == "location":
        # location_rel_obj = construct_location_rel(location_in_rel=True)
        question_1 = None
        if action in ["build", "copy", "spawn", "dig"]:
            question_1 = "Where should the " + optional_words + " happen?"
        elif action == "move":
            question_1 = "Where should the assistant move to"
        elif action == "dance":
            question_1 = "Where should the assistant dance"
        elif action == "otheraction":
            question_1 = "Give us more details about the location"

        QUESTION = [
            {"text": question_1, "key": "location", "radio": LOCATION_RADIO + LOCATION_REL_OBJECT},
            {
                "text": "If a number of steps is specified, how many ?",
                "key": "location.steps",
                "span": True,
                "optional": True,
                "tooltip": "e.g. in 'make a square 5 steps behind that' select '5'",
            },
        ]
        return QUESTION
    elif child == "tag_val":
        QUESTION = {
            "text": "Click on options below to determine the intent of the text.",
            "key": "memory_data",
            "tooltip": "e.g. in 'good job' click on 'Feedback to the assistant'",
            "add_radio_other": False,
            "radio": [
                {
                    "text": "Feedback to the assistant",
                    "key": "memory_type.reward",
                    "tooltip": "e.g. select for 'that was nice' or 'no that's wrong' ",
                    "next": [
                        {
                            "text": "Select the kind of feedback",
                            "key": "reward_value",
                            "add_radio_other": False,
                            "tooltip": "e.g. 'Positive feedback' for good things like 'you did a good job', 'that was a nice",
                            "radio": [
                                {
                                    "text": "Positive feedback",
                                    "key": "POSITIVE",
                                    "tooltip": "e.g. for good things like 'you did a good job', 'that was a nice'",
                                },
                                {
                                    "text": "Negative feedback",
                                    "key": "NEGATIVE",
                                    "tooltip": "e.g. for corrections like 'that was wrong', 'you failed'",
                                },
                            ],
                        }
                    ],
                },
                {
                    "text": "To assign tag, name or description",
                    "key": "memory_type.triple",
                    "tooltip": "e.g. 'that looks nice', 'tag the house as bright' etc",
                    "radio": [
                        {
                            "text": "The highlighted word(s) is a kind of colour",
                            "key": "has_colour",
                        },
                        {"text": "The highlighted word(s) represents size", "key": "has_size"},
                        {"text": "The highlighted word(s) is something else", "key": "has_tag"},
                    ],
                },
            ],
        }
    elif child == "reference_object" or (child == "filters" and action == "tag"):
        # location_rel_obj = construct_location_rel(location_in_rel=False)
        word = ""
        if action == "otheraction":
            question_1 = "There are words or pronouns that refer to the object (e.g. 'this', 'that', 'these', 'those', 'it' etc)"
            question_2 = "What is the name of the reference object"
        else:
            if action == "copy":
                word = "copied"
            elif action == "freebuild":
                word = "completed"
            elif action == "destroy":
                word = "destroyed"
            elif action == "fill":
                word = "filled"
            elif action == "spawn":
                word = "spawned"
            elif action == "tag":
                word = "tagged"

            question_1 = (
                "There are words or pronouns that refer to the object to be "
                + word
                + " (e.g. 'this', 'that', 'these', 'those', 'it' etc)"
            )
            question_2 = "What is the name of the object that should be " + word + "?"

        QUESTION = [
            {
                "text": "Click on all mentioned properties of the object in highlighted text.",
                "key": "reference_object",
                "checkbox": True,
                "tooltip": "e.g. in 'destroy the blue square' click on 'Name' as well as 'Colour' since both are specified in 'blue square'",
                "add_radio_other": False,
                "radio": [
                    {
                        "text": "Name",
                        "key": "name_check",
                        "tooltip": "Select this if the name / word for the object is mentioned",
                        "next": [{"text": question_2, "key": "has_name", "span": True}],
                    },
                    {
                        "text": question_1,
                        "key": "contains_coreference.yes",
                        "tooltip": "e.g. 'this', 'that', 'these', 'those', 'it' etc",
                        "add_radio_other": False,
                    },
                    {
                        "text": "The building material",
                        "key": "block_type_check",
                        "tooltip": "Select this if the building material of the object is mentioned",
                        "next": [
                            {
                                "text": "What is the building material? Select all words.",
                                "key": "has_block_type",
                                "span": True,
                                "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'glass'",
                            }
                        ],
                    },
                    {
                        "text": "Colour",
                        "key": "colour_check",
                        "tooltip": "Select this if the colour of the object is specified",
                        "next": [
                            {
                                "text": "What is the color?",
                                "key": "has_colour",
                                "span": True,
                                "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'blue'",
                            }
                        ],
                    },
                    {
                        "text": "Abstract/non-numeric size (e.g. 'big', 'small', etc.)",
                        "key": "size_check",
                        "tooltip": "Select this if the abstract/non-numeric size of the object is specified",
                        "next": [
                            {
                                "text": "What is the size?",
                                "key": "has_size",
                                "span": True,
                                "tooltip": "e.g. in 'destroy the tiny blue glass cube' select 'tiny'",
                            }
                        ],
                    },
                    {
                        "text": "Height",
                        "key": "height_check",
                        "tooltip": "Select this if the height is explicitly specified",
                        "next": [
                            {
                                "text": "What is the height?",
                                "key": "has_height",
                                "span": True,
                                "tooltip": "e.g. in 'complete the 20 blocks high tower' select '20'",
                            }
                        ],
                    },
                    {
                        "text": "Length",
                        "key": "length_check",
                        "tooltip": "Select this if the length is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for length.",
                                "key": "has_length",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 5 feet by 5 feet hole here' select '5'",
                            }
                        ],
                    },
                    {
                        "text": "Width",
                        "key": "width_check",
                        "tooltip": "Select this if the width is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for width",
                                "key": "has_width",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 2 by 3 hole there' select '3'",
                            }
                        ],
                    },
                    {
                        "text": "Depth",
                        "key": "depth_check",
                        "tooltip": "Select this if the depth is explicitly specified",
                        "next": [
                            {
                                "text": "Select all number words for depth",
                                "key": "has_depth",
                                "span": True,
                                "tooltip": "e.g. in 'dig a 1 x 2 x 3 pool' select '3'",
                            }
                        ],
                    },
                    {
                        "text": "Some other property not mentioned above",
                        "key": "tag_check",
                        "tooltip": "Select this if any property not explicitly mentioned above is given",
                        "next": [
                            {
                                "text": "Select all words for this property",
                                "key": "has_tag",
                                "span": True,
                                "tooltip": "e.g. in 'make a bright cabin' select 'bright'",
                            }
                        ],
                    },
                ],
            },
            {
                "text": "Is the location of the reference object mentioned ? Select all words.",
                "key": "reference_object.location",
                # "span": True,
                "tooltip": "e.g. in 'destroy the house behind the tree' select 'behind the tree'",
                "radio": LOCATION_RADIO + LOCATION_REL_OBJECT
                # [
                #     {
                #         "text": "Not specified",
                #         "key": None,
                #         "tooltip": "The location information is missing.",
                #     },
                #     {
                #         "text": "The location is represented using an indefinite noun like 'there' or 'over here'",
                #         "key": "CONTAINS_COREFERENCE",
                #         "tooltip": "e.g. 'there', 'here', 'over there' etc",
                #     },
                #     {
                #         "text": "Exact coordinates are given",
                #         "key": "coordinates_check",
                #         "tooltip": "Exact coordinates are specified.",
                #         "next": [
                #             {
                #                 "text": "Click on all words representing the coordinates",
                #                 "key": "yes.coordinates",
                #                 "span": True,
                #                 "tooltip": "e.g. in 'make a box at 4 , 5 , 6' select all: '4 , 5 , 6'",
                #             }
                #         ],
                #     },
                #     {
                #         "text": "Where the speaker is looking",
                #         "key": "SPEAKER_LOOK",
                #         "tooltip": "e.g. 'where I am looking'",
                #     },
                #     {
                #         "text": "Where the speaker is standing",
                #         "key": "SPEAKER_POS",
                #         "tooltip": "e.g. 'by me', 'where I am', 'where I am standing'",
                #     },
                #     {
                #         "text": "Where the assistant is standing",
                #         "key": "AGENT_POS",
                #         "tooltip": "e.g. 'by you', 'where you are', 'where you are standing'",
                #     },
                # ],
            },
        ]
        return QUESTION

    return QUESTION
