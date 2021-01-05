# 'reference_object', 'spawn'
# 'reference_object', 'destroy'
# 'reference_object', fill
# 'reference_object', 'OtherAction'
# 'reference_object', 'copy'

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
        {"text": "The time when this was created", "key": "BORN_TIME"},
        {"text": "The time when this was last modified or changed", "key": "MODIFY_TIME"},
        {"text": "The time when this was last visited.", "key": "VISIT_TIME"},
        {"text": "Relative direction or positioning", "key": "RELATIVE_DIRECTION"},
        {"text": "Number of blocks", "key": "NUM_BLOCKS"},
        {"text": "Distance to / from a given location", "key": "DISTANCE_TO"},
    ],
}

RANKED_OPTIONS = [QUANTITY_OPTIONS, ORDINAL_OPTIONS]

FIXED_OPTIONS = [QUANTITY_OPTIONS, NUMBER_OPTIONS]


def get_questions(child, action, ref_obj_child, optional_words=None):
    QUESTION = None
    # only handle reference objects or tag filters
    if (child != "reference_object" and child != "receiver_reference_object") and not (
        child == "filters" and action == "tag"
    ):
        return QUESTION

    # level one, get children values, spans of comparison and details of location
    if ref_obj_child == "":
        word = None
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

            question_1 = "There are words or pronouns that refer to the given object (e.g. 'this', 'that', 'these', 'those', 'it' etc)"
            question_2 = "What is the name of this object"
            if word is not None:
                question_1 = (
                    "There are words or pronouns that refer to the object to be "
                    + word
                    + " (e.g. 'this', 'that', 'these', 'those', 'it' etc)"
                )
                question_2 = "What is the name of the object that should be " + word + "?"

        QUESTION = [
            # # Table name
            # {
            #     "text": "What is the category of the highlighted object ?",
            #     "key": "reference_object.table_name",
            #     "tooltip": "e.g. in 'go to the house farthest from here' select 'A Physical object or structure'",
            #     "add_radio_other": False,
            #     "radio": [
            #         {
            #             "text": "A Physical object or structure",
            #             "key": "BlockObject",
            #             "tooltip": "e.g. 'house', 'square', 'cube', 'building' etc",
            #         },
            #         {
            #             "text": "An animal or moving entity",
            #             "key": "Mob",
            #             "tooltip": "e.g. 'cow', 'pig', 'sheep' etc",
            #         },
            #         #  More memory object types. here
            #     ],
            # },
            # the where clause
            {
                "text": "Select all other given properties of the object or person in highlighted text.",
                "key": "reference_object",
                "checkbox": True,
                "tooltip": "e.g. in 'destroy the blue square' click on 'Name' as well as 'Colour' since both are specified in 'blue square'",
                "add_radio_other": False,
                "radio": [
                    {
                        "text": "The entity being referred to is the Speaker",
                        "key": "special_reference.SPEAKER",
                        "tooltip": "e.g. in 'bring the book to me' select this option",
                    },
                    # Do we even need this ?
                    {
                        "text": "The entity being referred to is the assistant",
                        "key": "special_reference.AGENT",
                        "tooltip": "e.g. in 'get the cup to yourself' select this option",
                    },
                    {
                        "text": "Name",
                        "key": "name_check",
                        "tooltip": "Select this if the name / word for the object is mentioned",
                        "next": [
                            {
                                "text": question_2,
                                "key": "has_name",
                                "span": True,
                                "tooltip": "e.g. in 'destroy the cube' select 'cube'",
                            }
                        ],
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
                                "text": "What is the colour?",
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
                        "text": "This was a change caused by the person giving the command",
                        "key": "author.SPEAKER",
                        "tooltip": "The change was caused by the user e.g. 'go to the house that I built'",
                    },
                    {
                        "text": "This was a change caused by the assistant",
                        "key": "author.AGENT",
                        "tooltip": "The change was caused by the assistant e.g. 'go to the house you built'",
                    },
                    {
                        "text": "This was a change caused by another player",
                        "key": "author.PLAYER",
                        "tooltip": "The changed was caused by another player e.g. 'go to where that player was 2 minutes ago'",
                    },
                    {
                        "text": "Some property of this object is being compared or ranked",
                        "key": "arg_check",
                        "tooltip": "there is some form of comparison or ranking e.g. 'go to the farthest house from here', 'destroy the rightmost cube'",
                        "next": [
                            {
                                "text": "Select all words for the comparison and the type",
                                "key": "comparison",
                                "span": True,
                                "tooltip": "e.g. in 'fill the third black hole on your right' select 'third on your right'",
                            }
                        ],
                    },
                    {
                        "text": "Some other property not covered by anything above",
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
                "text": "Is the location of the reference object mentioned ?",
                "key": "reference_object.location",
                "tooltip": "e.g. in 'destroy the house behind the tree' select 'behind the tree'",
                "radio": LOCATION_RADIO + LOCATION_REL_OBJECT,
            }
            # NOTE: if get location span instead :
            # {
            #     "text": "Is the location of the reference object mentioned ? Select all words.",
            #     "key": "reference_object.location",
            #     "tooltip": "e.g. in 'destroy the house behind the tree' select 'behind the tree'",
            #     "span": True,
            # },
        ]
        return QUESTION

    return QUESTION
