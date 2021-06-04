import os
import json
import pkg_resources


def get_safety_words():
    """
    Extract the set of safety words from safety file
    and return
    """
    safety_words = set()
    safety_words_path = "{}/{}".format(
        pkg_resources.resource_filename("droidlet.documents", "internal"), "safety.txt"
    )
    if os.path.isfile(safety_words_path):
        """Read a set of safety words to prevent abuse."""
        with open(safety_words_path) as f:
            for l in f.readlines():
                w = l.strip("\n").lower()
                if w != "" and w[0] != "<" and w[0] != "#":
                    safety_words.add(w)
    return safety_words


def get_greetings(ground_truth_data_dir):
    # Load greetings
    greetings_path = ground_truth_data_dir + "greetings.json"
    greetings_map = {"hello": ["hi", "hello", "hey"], "goodbye": ["bye"]}
    if os.path.isfile(greetings_path):
        with open(greetings_path) as fd:
            greetings_map = json.load(fd)
    return greetings_map


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

