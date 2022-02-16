import os
import json


def get_ground_truth(no_ground_truth, ground_truth_data_dir):
    # Load all ground truth commands and their parses
    ground_truth_actions = {}
    if not no_ground_truth:
        if os.path.isdir(ground_truth_data_dir):
            gt_data_directory = ground_truth_data_dir + "datasets/"
            for (dirpath, dirnames, filenames) in os.walk(gt_data_directory):
                for f_name in filenames:
                    file = gt_data_directory + f_name
                    with open(file) as f:
                        for line in f.readlines():
                            text, logical_form = line.strip().split("|")
                            clean_text = text.strip('"')
                            ground_truth_actions[clean_text] = json.loads(logical_form)

    return ground_truth_actions

