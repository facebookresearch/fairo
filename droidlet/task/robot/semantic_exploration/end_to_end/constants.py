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
    56: 1,  # chair
    57: 6,  # couch
    58: 9,  # potted plant
    59: 7,  # bed
    61: 11,  # toilet
    62: 14,  # tv
}
