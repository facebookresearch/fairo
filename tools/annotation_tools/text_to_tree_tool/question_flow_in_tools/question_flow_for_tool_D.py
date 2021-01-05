# 'reference_object', 'comparison'
# 'reference_object', 'destroy'
# 'reference_object', fill
# 'reference_object', 'OtherAction'
# 'reference_object', 'copy'
# TODO: check if location can be integrated here ?
# TODO: check if filters can be integrated here

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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the first relative object?",
                            #     "key": "reference_object_1.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
                            {
                                "text": "Click on all words specifying the second object / area relative to which the location is given",
                                "key": "reference_object_2.has_name",
                                "tooltip": "e.g. in 'make 5 copies between the car and the house' select 'house'",
                                "span": True,
                            },
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the second relative object?",
                            #     "key": "reference_object_2.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
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
                            # {
                            #     "text": "Are there indefinite nouns or pronouns specifying the relative object?",
                            #     "key": "reference_object.contains_coreference",
                            #     "tooltip": "e.g. 'to the right of this', 'near that', 'behind these', 'next to those', 'underneath it' etc",
                            #     "add_radio_other": False,
                            #     "radio": [
                            #         {"text": "Yes", "key": "yes"},
                            #         {"text": "No", "key": "no"},
                            #     ],
                            # },
                        ],
                    },
                ],
            }
        ],
    }
]

RELATIVE_DIRECTION_LIST = [
    {
        "text": "Select the value of the relative direction",
        "key": "relative_direction",
        "radio": [
            {"text": "Left or towards the west direction", "key": "LEFT"},
            {"text": "Right or towards the east direction", "key": "RIGHT"},
            {"text": "Above or towards the north direction", "key": "UP"},
            {"text": "Below or towards the south direction", "key": "DOWN"},
            {"text": "In front", "key": "FRONT"},
            {"text": "Behind", "key": "BACK"},
            {"text": "Away from", "key": "AWAY"},
            {"text": "Inside", "key": "INSIDE"},
            {"text": "Outside", "key": "OUTSIDE"},
            {"text": "Between two object(s) / area(s)", "key": "BETWEEN"},
            {"text": "Nearby or close to", "key": "NEAR"},
            {"text": "Around", "key": "AROUND"},
            {"text": "Exactly at", "key": "EXACT"},
        ],
    }
]

ORDINAL_OPTIONS = {
    "text": "What is the position of the property or measure in the ranked list",
    "key": "ordinal",
    "add_radio_other": False,
    "radio": [
        {"text": "First / top of the list", "key": "FIRST"},
        {"text": "Second in the ordered list", "key": "SECOND"},
        {"text": "Third in the ordered list", "key": "THIRD"},
        {
            "text": "Some other rank in the list",
            "key": "ordinal_other",
            "tooltip": "e.g. 'destroy the fifth object to your right'",
            "next": [
                {
                    "text": "Select words in the text that represent the rank number",
                    "tooltip": "Select 'fifth' in 'destroy the fifth object to your right'",
                    "key": "ordinal_span",
                    "span": True,
                }
            ],
        },
    ],
}

NUMBER_OPTIONS = {
    "text": "Select words for the number being compared against",
    "key": "number",
    "span": True,
}

QUANTITY_OPTIONS = {
    "text": "Select the property or measure of what's being compared.",
    "key": "quantity",
    "add_radio_other": False,
    "radio": [
        {"text": "height", "key": "HEIGHT"},
        {"text": "width", "key": "WIDTH"},
        {"text": "depth", "key": "DEPTH"},
        {"text": "The time when this was created", "key": "BORN_TIME"},
        {"text": "The time when this was last modified or changed", "key": "MODIFY_TIME"},
        {"text": "The time when this was last visited.", "key": "VISIT_TIME"},
        {
            "text": "Relative direction or positioning from agent",
            "key": "RELATIVE_DIRECTION",
            "next": RELATIVE_DIRECTION_LIST,
        },
        {
            "text": "Number of blocks",
            "key": "NUM_BLOCKS",
            "next": [
                {
                    "text": "Select all properties of the block type.",
                    "key": "block_filters",
                    "add_radio_other": False,
                    "checkbox": True,
                    "radio": [
                        {
                            "text": "Colour",
                            "key": "colour_check",
                            "tooltip": "Select this if the colour of the object is specified",
                            "next": [
                                {
                                    "text": "What is the colour?",
                                    "key": "has_colour",
                                    "span": True,
                                    "tooltip": "e.g. in 'destroy the cube with most red blocks' select 'red'",
                                }
                            ],
                        },
                        {
                            "text": "The block type material",
                            "key": "block_type_check",
                            "tooltip": "Select this if the material of the block is mentioned",
                            "next": [
                                {
                                    "text": "What is the block material? Select all words.",
                                    "key": "has_block_type",
                                    "span": True,
                                    "tooltip": "e.g. in 'go to the house with most stone blocks' select 'stone'",
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        {
            "text": "Distance to / from a given location",
            "key": "distance_to",
            "tooltip": "e.g. 'go to the blue circle farthest from me'",
            "next": [
                {
                    "text": "Tell us about the type of location from which distance is being computed",
                    "key": "location",
                    "tooltip": "e.g. in 'destroy the house behind the tree' select 'behind the tree'",
                    "radio": LOCATION_RADIO + LOCATION_REL_OBJECT,
                }
            ],
        },
    ],
}


def get_questions(child, ref_obj_child):
    QUESTION = None

    # only handle comparison in this tool
    if ref_obj_child != "comparison":
        return QUESTION

    QUESTION = [
        QUANTITY_OPTIONS,
        {
            "text": "Select the kind of comparison of the value of a given property",
            "key": "arg_check_type",
            "add_radio_other": False,
            "radio": [
                {
                    "text": "The value is ranked.",
                    "key": "ranked",
                    "tooltip": "e.g. 'go to the third house from here', 'destroy the cube closest to you'",
                    "next": [
                        {
                            "text": "Select the kind of measure",
                            "key": "measure_check",
                            "add_radio_other": False,
                            "radio": [
                                {
                                    "text": "The value is minimum / smallest / closest measure of some kind.",
                                    "key": "argmin",
                                    "tooltip": "minimum first e.g. in 'destroy the red thing closest to me', 'fill the third hole to my right'",
                                    "next": [ORDINAL_OPTIONS],
                                },
                                {
                                    "text": "The value is maximum / largest / farthest measure of some kind.",
                                    "key": "argmax",
                                    "tooltip": "maximum first e.g in 'go to the house farthest from me', 'copy the second largest house'",
                                    "next": [ORDINAL_OPTIONS],
                                },
                            ],
                        }
                    ],
                },
                {
                    "text": "The value is being compared to a fixed number.",
                    "key": "fixed",
                    "tooltip": "e.g. 'fill the hole that is more than 10 block deep', 'destroy the thing you made 5 minutes back'",
                    "next": [
                        {
                            "text": "Select the kind of comparison",
                            "key": "measure_check",
                            "add_radio_other": False,
                            "radio": [
                                {
                                    "text": "The value is greater than some number",
                                    "key": "greater_than",
                                    "tooltip": "e.g. 'fill the hole that is more than 2 blocks wide'",
                                    "next": [NUMBER_OPTIONS],
                                },
                                {
                                    "text": "The value is smaller than some number",
                                    "key": "less_than",
                                    "tooltip": "e.g. 'destroy the house that has less than 3 windows'",
                                    "next": [NUMBER_OPTIONS],
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    ]
    return QUESTION
