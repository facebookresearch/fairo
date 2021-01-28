"""
Preprocess

This script reads text files from a directory and creates a JSON map of commands
to action dictionaries.
"""
import glob
import ast
import json

# Dictionary of all commands and parse trees
all_commands = {}

ANNOTATIONS_DIR_PATH = "/Users/rebeccaqian/datasets/autocomplete/"
JSON_OUT_PATH = "/Users/rebeccaqian/datasets/autocomplete/annotations.json"

for file_path in glob.glob(ANNOTATIONS_DIR_PATH + "*.txt"):
    with open(file_path) as fd:
        data = fd.readlines()

    for line in data:
        if "|" in line:
            command, action_dict = line.split("|")
            action_dict = ast.literal_eval(action_dict)
        else:
            command = line
            action_dict = {}
        # print(data)
        if command not in all_commands:
            all_commands[command.strip()] = action_dict


# print(all_commands)
with open(JSON_OUT_PATH, "w") as fd:
    json.dump(all_commands, fd)