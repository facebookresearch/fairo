import os
import sys
import ast
import copy
import pprint as pp
import argparse
import json

"""Applies updates to annotated dataset following grammar changes.
"""

def read_file(file_path):
    with open(file_path) as fd:
        dataset = fd.readlines()
    return dataset

def update_tree():
    raise NotImplementedException

def traverse_tree(command: str, action_dict: dict):
    # if "action_sequence" not in action_dict:
    #     print(action_dict)
    #     return action_dict
    # for action in action_dict["action_sequence"]:
    traverse_subtree(command, action_dict)
    print("final tree:")
    pp.pprint(action_dict)
    print(action_dict)
    return action_dict

def get_span_range(text: str, command: str):
    index = command.find(text)
    if index == -1:
        return index
    else:
        words_arr = command.split(" ")
        # import ipdb; ipdb.set_trace()
        text_arr = text.split(" ")
        # start = words_arr
        for i in range(len(words_arr)):
            if words_arr[i] == text_arr[0]:
                words_arr_set = " ".join(words_arr[i:i+len(text_arr)])
                # import ipdb; ipdb.set_trace()

                if words_arr_set == text:
                    # import ipdb; ipdb.set_trace()
                    return [0, [i, i+len(text_arr) -1]]

        return -1
                # start_index = i 
            # if start_index:
            #     for j in range(text_arr):
    # words_arr = command.split(" ")
    # text_arr = text.split()

def traverse_subtree(command: str, action_dict: dict):
    for key, value in [x for x in action_dict.items()]:
        # import ipdb; ipdb.set_trace()
        if type(value) == dict:
            traverse_subtree(command, value)
        if type(value) == list:
            for ad in value:
                traverse_subtree(command, ad)
        if value == "":
            del action_dict[key]
        # if "text" in key:
        #     # import ipdb; ipdb.set_trace()
        #     index_range = get_span_range(value, command)
        #     if type(index_range) == list and type(index_range[0]) == int:
        #         # import ipdb; ipdb.set_trace()
        #         action_dict[key] = index_range

    return action_dict

def write_file(dataset, file_path):
    with open(file_path, "w") as fd:
        for line in dataset:
            fd.write(line + "\n")

if __name__ == "__main__":
    print("*** Applying grammar updates ***")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default="get_memory_command_dict_pairs.json")
    parser.add_argument("--dest_path", default="autocomplete_annotations.txt")
    args = parser.parse_args()
    # load the annotated dataset
    dataset = json.load(open(args.source_path))
    updated_dataset = []
    for command in dataset:
        action_dict = dataset[command]
        updated_tree = traverse_tree(command, action_dict)
        updated_dataset.append("{}|{}".format(command, str(updated_tree)))
    write_file(updated_dataset, args.dest_path)
