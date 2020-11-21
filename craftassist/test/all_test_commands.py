"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from base_agent.dialogue_objects import SPEAKERLOOK, AGENTPOS
from copy import deepcopy

ATTRIBUTES = {
    "x": {"attribute": "x"},
    "distance from me": {
        "attribute": {
            "linear_extent": {
                "relative_direction": "AWAY",
                "source": {"special_reference": "SPEAKER"},
            }
        }
    },
    "create time": {"attribute": "BORN_TIME"},
    "number of blocks": {"num_blocks": {"block_filters": {}}},
    "number of blue blocks": {"num_blocks": {"block_filters": {"has_colour": "blue"}}},
}


FILTERS = {
    "that cow": {"has_name": "cow", "contains_coreference": "resolved", "location": SPEAKERLOOK},
    "that cube": {"has_name": "cube", "contains_coreference": "resolved", "location": SPEAKERLOOK},
    "a cow": {"has_name": "cow"},
    "a cube": {"has_name": "cube"},
    "where I am looking": {"location": SPEAKERLOOK},
    "my location": {"location": AGENTPOS},
    "number of blocks in blue cube": {
        "output": {"attribute": ATTRIBUTES["number of blocks"]},
        "has_name": "cube",
        "has_colour": "blue",
    },
}

REFERENCE_OBJECTS = {
    "where I am looking": {
        "filters": FILTERS["where I am looking"],
        "text_span": "where I'm looking",
    },
    "that cow": {"filters": FILTERS["that cow"]},
    "a cow": {"filters": FILTERS["a cow"]},
    "that cube": {"filters": FILTERS["that cube"]},
    "a cube": {"filters": FILTERS["a cube"]},
    "me": {"special_reference": "AGENT"},
}

ATTRIBUTES["distance from that cube"] = {
    "attribute": {
        "linear_extent": {"relative_direction": "AWAY", "source": REFERENCE_OBJECTS["that cube"]}
    }
}

# FIXME "built" should check for player made or agent made
FILTERS["the first thing that was built"] = {
    "argmin": {"ordinal": "FIRST", "quantity": ATTRIBUTES["create time"]},
    "has_tag": "_voxel_object",
}
FILTERS["the last thing that was built"] = {
    "argmax": {"ordinal": "FIRST", "quantity": ATTRIBUTES["create time"]},
    "has_tag": "_voxel_object",
}
FILTERS["number of blocks in the first thing built"] = {
    "output": {"attribute": ATTRIBUTES["number of blocks"]},
    "argmin": {"ordinal": "FIRST", "quantity": ATTRIBUTES["create time"]},
    "has_tag": "_voxel_object",
}
FILTERS["number of blocks in the second thing built"] = {
    "output": {"attribute": ATTRIBUTES["number of blocks"]},
    "argmin": {"ordinal": "SECOND", "quantity": ATTRIBUTES["create time"]},
    "has_tag": "_voxel_object",
}
FILTERS["number of blocks in the last thing built"] = {
    "output": {"attribute": ATTRIBUTES["number of blocks"]},
    "argmax": {"ordinal": "FIRST", "quantity": ATTRIBUTES["create time"]},
    "has_tag": "_voxel_object",
}


INTERPRETER_POSSIBLE_ACTIONS = {
    "destroy_speaker_look": {
        "action_type": "DESTROY",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
    "spawn_5_sheep": {
        "action_type": "SPAWN",
        "reference_object": {"filters": {"has_name": "sheep"}, "text_span": "sheep"},
        "repeat": {"repeat_key": "FOR", "repeat_count": "5"},
    },
    "copy_speaker_look_to_agent_pos": {
        "action_type": "BUILD",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
        "location": {
            "reference_object": {"special_reference": "AGENT"},
            "text_span": "where I am",
        },
    },
    "build_small_sphere": {
        "action_type": "BUILD",
        "schematic": {"has_name": "sphere", "has_size": "small", "text_span": "small sphere"},
    },
    "build_1x1x1_cube": {
        "action_type": "BUILD",
        "schematic": {"has_name": "cube", "has_size": "1 x 1 x 1", "text_span": "1 x 1 x 1 cube"},
    },
    "move_speaker_pos": {
        "action_type": "MOVE",
        "location": {"reference_object": {"special_reference": "SPEAKER"}, "text_span": "to me"},
    },
    "build_diamond": {
        "action_type": "BUILD",
        "schematic": {"has_name": "diamond", "text_span": "diamond"},
    },
    "build_gold_cube": {
        "action_type": "BUILD",
        "schematic": {"has_block_type": "gold", "has_name": "cube", "text_span": "gold cube"},
    },
    "build_red_cube": {
        "action_type": "BUILD",
        "location": {"reference_object": {"special_reference": "SPEAKER_LOOK"}},
        "schematic": {"has_colour": "red", "has_name": "cube", "text_span": "red cube"},
    },
    "destroy_red_cube": {
        "action_type": "DESTROY",
        "reference_object": {
            "filters": {"has_name": "cube", "has_colour": "red"},
            "text_span": "red cube",
        },
    },
    "fill_all_holes_speaker_look": {
        "action_type": "FILL",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
        "repeat": {"repeat_key": "ALL"},
    },
    "go_to_tree": {
        "action_type": "MOVE",
        "location": {"reference_object": {"filters": {"has_name": "tree"}}, "text_span": "tree"},
    },
    "build_square_height_1": {
        "action_type": "BUILD",
        "schematic": {"has_name": "square", "has_height": "1", "text_span": "square height 1"},
    },
    "stop": {"action_type": "STOP"},
    "fill_speaker_look": {
        "action_type": "FILL",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
    "fill_speaker_look_gold": {
        "action_type": "FILL",
        "has_block_type": "gold",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
}

BUILD_COMMANDS = {
    "build a gold cube at 0 66 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {"has_name": "cube", "has_block_type": "gold"},
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "0 66 0"}}
                },
            }
        ],
    },
    "build a small cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "BUILD", "schematic": {"has_name": "cube", "has_size": "small"}}
        ],
    },
    "build a circle to the left of the circle": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "location": {
                    "reference_object": {"filters": {"has_name": "circle"}},
                    "relative_direction": "LEFT",
                    "text_span": "to the left of the circle",
                },
                "schematic": {"has_name": "circle", "text_span": "circle"},
            }
        ],
    },
    "copy where I am looking to here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["copy_speaker_look_to_agent_pos"]],
    },
    "build a small sphere": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_small_sphere"]],
    },
    "build a 1x1x1 cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_1x1x1_cube"]],
    },
    "build a diamond": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_diamond"]],
    },
    "build a gold cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_gold_cube"]],
    },
    "build a 9 x 9 stone rectangle": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {
                    "has_block_type": "stone",
                    "has_name": "rectangle",
                    "has_height": "9",
                    "has_base": "9",  # has_base doesn't belong in "rectangle"
                    "text_span": "9 x 9 stone rectangle",
                },
            }
        ],
    },
    "build a square with height 1": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_square_height_1"]],
    },
    "build a red cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_red_cube"]],
    },
    "build a fluffy here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {"has_name": "fluffy"},
                "location": {"reference_object": {"special_reference": "AGENT"}},
            }
        ],
    },
}

SPAWN_COMMANDS = {
    "spawn 5 sheep": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["spawn_5_sheep"]],
    }
}

DESTROY_COMMANDS = {
    "destroy it": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "DESTROY",
                "reference_object": {"filters": {"contains_coreference": "yes"}},
            }
        ],
    },
    "destroy where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["destroy_speaker_look"]],
    },
    "destroy the red cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["destroy_red_cube"]],
    },
    "destroy everything": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "reference_object": {
                    "repeat": {"repeat_key": "ALL"},
                    "filters": {},
                    "text_span": "everything",
                },
                "action_type": "DESTROY",
            }
        ],
    },
    "destroy the fluff thing": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "DESTROY", "reference_object": {"filters": {"has_tag": "fluff"}}}
        ],
    },
    "destroy the fluffy object": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "DESTROY", "reference_object": {"filters": {"has_tag": "fluffy"}}}
        ],
    },
}

MOVE_COMMANDS = {
    "move to 42 65 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "42 65 0"}}
                },
            }
        ],
    },
    "move to 0 63 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "0 63 0"}}
                },
            }
        ],
    },
    "move to -7 63 -8": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "-7 63 -8"}},
                    "text_span": "-7 63 -8",
                },
            }
        ],
    },
    "go between the cubes": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"filters": {"has_name": "cube"}},
                    "relative_direction": "BETWEEN",
                    "text_span": "between the cubes",
                },
            }
        ],
    },
    "move here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["move_speaker_pos"]],
    },
    "go to the tree": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["go_to_tree"]],
    },
    "move to 20 63 20": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "20 63 20"}},
                    "text_span": "20 63 20",
                },
            }
        ],
    },
}

FILL_COMMANDS = {
    "fill where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_speaker_look"]],
    },
    "fill where I am looking with gold": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_speaker_look_gold"]],
    },
    "fill all holes where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_all_holes_speaker_look"]],
    },
}

DANCE_COMMANDS = {
    "dance": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [{"action_type": "DANCE", "dance_type": {"dance_type_span": "dance"}}],
    }
}

COMBINED_COMMANDS = {
    "build a small sphere then move here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            INTERPRETER_POSSIBLE_ACTIONS["build_small_sphere"],
            INTERPRETER_POSSIBLE_ACTIONS["move_speaker_pos"],
        ],
    },
    "copy where I am looking to here then build a 1x1x1 cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            INTERPRETER_POSSIBLE_ACTIONS["copy_speaker_look_to_agent_pos"],
            INTERPRETER_POSSIBLE_ACTIONS["build_1x1x1_cube"],
        ],
    },
    "move to 3 63 2 then 7 63 7": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "3 63 2"}},
                    "text_span": "3 63 2",
                },
            },
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "7 63 7"}},
                    "text_span": "7 63 7",
                },
            },
        ],
    },
}

# maybe fixme ACTION_NAME-->NAME ?
GET_MEMORY_COMMANDS = {
    "what is where I am looking": {
        "dialogue_type": "GET_MEMORY",
        "filters": {
            "memory_type": "REFERENCE_OBJECT",
            "location": SPEAKERLOOK,
            "output": {"attribute": "NAME"},
        },
    },
    "what are you doing": {
        "dialogue_type": "GET_MEMORY",
        "filters": {
            "memory_type": {"action_type": "NULL"},
            "output": {"attribute": "ACTION_NAME"},
        },
    },
    "what are you building": {
        "dialogue_type": "GET_MEMORY",
        "filters": {
            "memory_type": {"action_type": "BUILD"},
            "output": {"attribute": "ACTION_REFERENCE_OBJECT_NAME"},
        },
    },
    "where are you going": {
        "dialogue_type": "GET_MEMORY",
        "filters": {
            "memory_type": {"action_type": "MOVE"},
            "output": {"attribute": "MOVE_TARGET"},
        },
    },
    "where are you": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"memory_type": "AGENT", "output": {"attribute": "LOCATION"}},
    },
}

PUT_MEMORY_COMMANDS = {
    "that is fluff": {
        "dialogue_type": "PUT_MEMORY",
        "filters": {"location": SPEAKERLOOK},
        "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluff"}},
    },
    "good job": {
        "dialogue_type": "PUT_MEMORY",
        "upsert": {"memory_data": {"memory_type": "REWARD", "reward_value": "POSITIVE"}},
    },
    "that is fluffy": {
        "dialogue_type": "PUT_MEMORY",
        "filters": {"location": SPEAKERLOOK},
        "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluffy"}},
    },
}


def append_output(filt, output):
    new_filt = deepcopy(filt)
    new_filt["output"] = output
    return new_filt


CONDITIONS = {
    "a cow has x greater than 5": {
        "condition_type": "COMPARATOR",
        "condition": {
            "input_left": {"value_extractor": append_output(FILTERS["a cow"], ATTRIBUTES["x"])},
            "comparison_type": "GREATER_THAN",
            "input_right": {"value_extractor": "5"},
        },
    },
    "that cow has x greater than 5": {
        "condition_type": "COMPARATOR",
        "condition": {
            "input_left": {"value_extractor": append_output(FILTERS["that cow"], ATTRIBUTES["x"])},
            "comparison_type": "GREATER_THAN",
            "input_right": {"value_extractor": "5"},
        },
    },
    "that cow is closer than 2 steps to me": {
        "condition_type": "COMPARATOR",
        "condition": {
            "input_left": {
                "value_extractor": append_output(
                    FILTERS["that cow"], ATTRIBUTES["distance from me"]
                )
            },
            "comparison_type": "LESS_THAN",
            "input_right": {"value_extractor": "2"},
        },
    },
    "2 minutes": {
        "condition_type": "TIME",
        "condition": {
            "comparator": {
                "comparison_measure": "minutes",
                "input_left": {"value_extractor": "NULL"},
                "comparison_type": "GREATER_THAN",
                "input_right": {"value_extractor": "2"},
            }
        },
    },
    "18 seconds": {
        "condition_type": "TIME",
        "condition": {
            "comparator": {
                "comparison_measure": "seconds",
                "input_left": {"value_extractor": "NULL"},
                "comparison_type": "GREATER_THAN",
                "input_right": {"value_extractor": "18"},
            }
        },
    },
}

CONDITIONS["18 seconds after that cow has x greater than 5"] = {
    "condition_type": "TIME",
    "event": CONDITIONS["that cow has x greater than 5"],
    "condition": {
        "comparator": {
            "comparison_measure": "seconds",
            "input_left": {"value_extractor": "NULL"},
            "comparison_type": "GREATER_THAN",
            "input_right": {"value_extractor": "18"},
        }
    },
}


STOP_CONDITION_COMMANDS = {
    "go left until that cow is closer than 2 steps to me": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                },
                "stop_condition": CONDITIONS["that cow is closer than 2 steps to me"],
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the cow for 2 minutes": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "cow"}}},
                "stop_condition": CONDITIONS["2 minutes"],
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the cow for 18 seconds": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "cow"}}},
                "stop_condition": CONDITIONS["18 seconds"],
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the cow for 18 seconds after it has x greater than 5": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "cow"}}},
                "stop_condition": CONDITIONS["18 seconds after that cow has x greater than 5"],
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the cow until it has x greater than 5": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "cow"}}},
                "stop_condition": CONDITIONS["that cow has x greater than 5"],
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
}


OTHER_COMMANDS = {
    "the weather is good": {"dialogue_type": "NOOP"},
    "stop": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["stop"]],
    },
    "undo": {"dialogue_type": "HUMAN_GIVE_COMMAND", "action_sequence": [{"action_type": "UNDO"}]},
}

GROUND_TRUTH_PARSES = {
    "go to the gray chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"filters": {"has_colour": "gray", "has_name": "chair"}}
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go to the chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "chair"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward 0.2 meters": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "0.2",
                    "has_measure": "meters",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward one meter": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "one",
                    "has_measure": "meter",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go left 3 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                    "steps": "3",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go left 3 meters": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                    "steps": "3",
                    "has_measure": "meters",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward 1 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "1",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go back 1 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "BACK",
                    "steps": "1",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right 90 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "-90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn left 90 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right 180 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "-180"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "-90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "look at where I am pointing": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {
                    "look_turn": {
                        "location": {"reference_object": {"special_reference": "SPEAKER_LOOK"}}
                    }
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "wave": {
        "action_sequence": [{"action_type": "DANCE", "dance_type": {"dance_type_name": "wave"}}],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "chair"}}},
                "stop_condition": {"condition_type": "NEVER"},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "find Laurens": {
        "action_sequence": [
            {"action_type": "SCOUT", "reference_object": {"filters": {"has_name": "Laurens"}}}
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "bring the cup to Mary": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {"reference_object": {"filters": {"has_name": "Mary"}}},
                "reference_object": {"filters": {"has_name": "cup"}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go get me lunch": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {"reference_object": {"special_reference": "SPEAKER"}},
                "reference_object": {"filters": {"has_name": "lunch"}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
}
