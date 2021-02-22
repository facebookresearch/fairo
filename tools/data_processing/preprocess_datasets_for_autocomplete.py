"""
Preprocessing for Autocomplete Annotation Tool.

This script reads text files from a directory and creates a JSON map of commands
to action dictionaries.
"""
import glob
import ast
import json
import argparse


def create_JSON_from_txt(annotations_dir_path):
    # Dictionary of all commands and parse trees
    all_commands = {}
    output_JSON_commands = {}
    commands_to_annotate = open(args.commands_path).read().splitlines()

    for file_path in glob.glob(annotations_dir_path + "*.txt"):
        # Skip templated generations
        if "templated" in file_path:
            continue
        
        with open(file_path) as fd:
            data = fd.readlines()

        for line in data:
            if "|" in line:
                command, action_dict = line.strip().split("|")
                action_dict = json.loads(action_dict)
            else:
                command = line
                action_dict = {}
            if command.strip() not in all_commands:
                all_commands[command.strip()] = action_dict
        
    # Remove the commands that are not needing to be annotated
    for command in all_commands:
        if command in commands_to_annotate:
            output_JSON_commands[command] = all_commands[command]
        
    return output_JSON_commands


def write_JSON(json_out_path, all_commands):
    with open(json_out_path, "w") as fd:
        json.dump(all_commands, fd)

if __name__ == "__main__":
    print("*** Preparing data store for Autocomplete Tool ***")
    parser = argparse.ArgumentParser()
    # Default is assuming script is run from template tool backend
    parser.add_argument("--annotations_dir_path", default="../../../../craftassist/agent/datasets/full_data/")
    parser.add_argument("--commands_path", default="commands.txt")
    parser.add_argument("--json_out_path", default="command_dict_pairs.json")
    args = parser.parse_args()
    print("*** Loading data pairs from {} ***".format(args.annotations_dir_path))
    all_commands = create_JSON_from_txt(args.annotations_dir_path)
    write_JSON(args.json_out_path, all_commands)
    print("*** Wrote JSON to {} ***".format(args.json_out_path))