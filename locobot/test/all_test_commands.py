"""
Copyright (c) Facebook, Inc. and its affiliates.
"""


def command(d):
    if type(d) is list:
        return {"dialogue_type": "HUMAN_GIVE_COMMAND", "action_sequence": d}
    else:
        return {"dialogue_type": "HUMAN_GIVE_COMMAND", "action_sequence": [d]}


MOVE_COMMANDS = {
    "move_forward": command({"action_type": "MOVE", "location": {"relative_direction": "FRONT"}}),
    "stop": command({"action_type": "STOP"}),
    "move to -7 0 -8": command(
        {
            "action_type": "MOVE",
            "location": {
                "reference_object": {"special_reference": {"coordinates_span": "-7 0 -8"}},
                "text_span": "-7 0 -8",
            },
        }
    ),
    "move to 3 0 2 then 7 0 7": command(
        [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "3 0 2"}},
                    "text_span": "3 0 2",
                },
            },
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "7 0 7"}},
                    "text_span": "7 0 7",
                },
            },
        ]
    ),
    "go between the cubes": command(
        {
            "action_type": "MOVE",
            "location": {
                "reference_object": {
                    "filters": {"triples": [{"pred_text": "has_name", "obj_text": "cube"}]}
                },
                "relative_direction": "BETWEEN",
                "text_span": "between the cubes",
            },
        }
    ),
    "move here": command(
        {
            "action_type": "MOVE",
            "location": {
                "reference_object": {"special_reference": "SPEAKER"},
                "text_span": "to me",
            },
        }
    ),
    "look at the cube": command(
        {
            "action_type": "DANCE",
            "dance_type": {
                "look_turn": {
                    "location": {
                        "reference_object": {
                            "filters": {"triples": [{"pred_text": "has_name", "obj_text": "cube"}]}
                        },
                        "text_span": "cube",
                    }
                }
            },
        }
    ),
    "go to the cube": command(
        {
            "action_type": "MOVE",
            "location": {
                "reference_object": {
                    "filters": {"triples": [{"pred_text": "has_name", "obj_text": "cube"}]}
                },
                "text_span": "cube",
            },
        }
    ),
    "get the toy": command(
        {
            "action_type": "GET",
            "reference_object": {
                "filters": {"triples": [{"pred_text": "has_name", "obj_text": "toy"}]}
            },
        }
    ),
}

GROUND_TRUTH_PARSES = {
    "go to the gray chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {
                        "filters": {
                            "triples": [
                                {"pred_text": "has_colour", "obj_text": "gray"},
                                {"pred_text": "has_name", "obj_text": "chair"},
                            ]
                        }
                    }
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go to the chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {
                        "filters": {"triples": [{"pred_text": "has_name", "obj_text": "chair"}]}
                    }
                },
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
    "go left 4 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                    "steps": "4",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go right 3 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "RIGHT",
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
                "location": {
                    "reference_object": {
                        "filters": {"triples": [{"pred_text": "has_name", "obj_text": "chair"}]}
                    }
                },
                "stop_condition": {"condition_type": "NEVER"},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "find Laurens": {
        "action_sequence": [
            {
                "action_type": "SCOUT",
                "reference_object": {
                    "filters": {"triples": [{"pred_text": "has_name", "obj_text": "Laurens"}]}
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "bring the cup to Mary": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {
                    "reference_object": {
                        "filters": {"triples": [{"pred_text": "has_name", "obj_text": "Mary"}]}
                    }
                },
                "reference_object": {
                    "filters": {"triples": [{"pred_text": "has_name", "obj_text": "cup"}]}
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go get me lunch": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {"reference_object": {"special_reference": "SPEAKER"}},
                "reference_object": {
                    "filters": {"triples": [{"pred_text": "has_name", "obj_text": "lunch"}]}
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
}
