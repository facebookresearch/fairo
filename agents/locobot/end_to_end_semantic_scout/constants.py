coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "no-category": 15,
}

coco_id_to_goal_id = {
    0: 0,  # chair
    1: 5,  # couch
    2: 2,  # potted plant
    3: 1,  # bed
    4: 3,  # toilet
    5: 4,  # tv
}

# For visualization with colors we're used to
expected_categories_to_coco_categories = {
    1: 0,  # chair
    6: 1,  # couch
    9: 2,  # potted plant
    7: 3,  # bed
    11: 4,  # toilet
    14: 5,  # tv
}

# To replace the segmentation model trained on simulation data
# with a segmentation model trained on real-world data
detectron_categories_to_expected_categories = {
    # Goal categories: most important
    56: 1,  # chair (MP3D 3)
    57: 6,  # couch (MP3D 10)
    58: 9,  # plant (MP3D 14)
    59: 7,  # bed (MP3D 11)
    61: 11,  # toilet (MP3D 18)
    62: 14,  # tv (MP3D 22)
    # Other categories
    # 60: 2,  # dining table (MP3D 5)
    # ??: 3,  # picture (MP3D 6)
    # ??: 4,  # cabinet (MP3D 7)
    # ??: 5,  # cushion (MP3D 8)
    # ??: 8,  # chest_of_drawers (MP3D 13)
    # 71: 10,  # sink (MP3D 15)
    # ??: 12,  # stool (MP3D 19)
    # ??: 13,  # towel (MP3D 20)
    # ??: 15,  # shower (MP3D 23)
    # ??: 16,  # bathtub (MP3D 25)
    # ??: 17,  # counter (MP3D 26)
    # ??: 18,  # fireplace (MP3D 27)
    # ??: 19,  # gym_equipment (MP3D 33)
    # ??: 20,  # seating (MP3D 34)
    # ??: 21,  # clothes (MP3D 38)
}

mmdetection_categories_to_expected_categories = {
    0: 1,   # chair
    5: 6,   # couch
    8: 9,   # potted plant
    6: 7,   # bed
    10: 11,  # toilet
    13: 14,  # tv
}

coco_categories_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]

frame_color_palette = [
    *coco_categories_color_palette,
    1.0,
    1.0,
    1.0,  # no category
]
