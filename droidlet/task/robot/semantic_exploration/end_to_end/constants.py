mp3d_categories_id2name = {
    0: "void",
    1: "wall",
    2: "floor",
    3: "chair",
    4: "door",
    5: "table",
    6: "picture",
    7: "cabinet",
    8: "cushion",
    9: "window",
    10: "sofa",
    11: "bed",
    12: "curtain",
    13: "chest_of_drawers",
    14: "plant",
    15: "sink",
    16: "stairs",
    17: "ceiling",
    18: "toilet",
    19: "stool",
    20: "towel",
    21: "mirror",
    22: "tv_monitor",
    23: "shower",
    24: "column",
    25: "bathtub",
    26: "counter",
    27: "fireplace",
    28: "lighting",
    29: "beam",
    30: "railing",
    31: "shelving",
    32: "blinds",
    33: "gym_equipment",
    34: "seating",
    35: "board_panel",
    36: "furniture",
    37: "appliances",
    38: "clothes",
    39: "objects",
    40: "misc",
    41: "unlabeled",
}

detectron_categories_to_mp3d_categories = {
    # Most important: goal categories
    56: 3,  # chair
    57: 10,  # couch
    58: 14,  # potted plant
    59: 11,  # bed
    61: 18,  # toilet
    62: 22,  # tv
    # Other: add more as needed
    60: 5,  # dining table
    71: 15,  # sink
}
