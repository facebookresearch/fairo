import json
import os
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