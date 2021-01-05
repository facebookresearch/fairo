import os
import sys
import ast
import copy
import pprint as pp
import argparse

"""Applies updates to annotated dataset following grammar changes.
"""

def read_file(file_path):
    with open(file_path) as fd:
        dataset = fd.readlines()
    return dataset

def update_tree():
    raise NotImplementedException

def traverse_tree(action_dict: dict):
    if "action_sequence" not in action_dict:
        traverse_subtree(action_dict)
        return action_dict
    for action in action_dict["action_sequence"]:
        traverse_subtree(action)
    return action_dict

def traverse_subtree(action_dict: dict):
    for key, value in [x for x in action_dict.items()]:
        if type(value) == dict:
            traverse_subtree(value)
        if type(value) == list:
            if type(value[0]) == list:
                action_dict[key] = value[0]
            else:
                for ad in value:
                    traverse_subtree(ad)
        if key == "text_span":
            action_dict[key] = value[0]
        elif "has_" in key:
            if "triples" in action_dict:
                action_dict["triples"].append({"pred_text": key, "obj_text": action_dict[key]})
            else:
                action_dict["triples"] = [{"pred_text": key, "obj_text": action_dict[key]}]
            del action_dict[key]
    return action_dict

def write_file(dataset, file_path):
    with open(file_path, "w") as fd:
        for line in dataset:
            fd.write(line + "\n")

if __name__ == "__main__":
    print("*** Applying grammar updates ***")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default="craftassist/agent/datasets/full_data/templated.txt")
    parser.add_argument("--dest_path", default="craftassist/agent/datasets/full_data/templated.txt")
    args = parser.parse_args()
    # load the annotated dataset
    dataset = read_file(args.source_path)
    updated_dataset = []
    for row in dataset:
        command, action_dict = row.split("|")
        action_dict = ast.literal_eval(action_dict)
        updated_tree = traverse_tree(action_dict)
        updated_dataset.append('{}|{}'.format(command, str(updated_tree)))
    write_file(updated_dataset, args.dest_path)
