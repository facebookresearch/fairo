import glob
from random import shuffle
import re
import argparse

"""
Create static validation set, defaults to 2% of each data type
"""

DATA_CHECKPOINT = ""
COMMANDS_CHECKPOINT = ""
TRAIN_VALID_DATA_CHECKPOINT = ""


def write_data_chunk_to_file(data, file, commands_only=False):
    with open(file, "w") as fd:
        for line in data:
            if commands_only:
                command, action_dict = line.split("|")
                fd.write(command + "\n")
            else:
                fd.write(line)


def load_file_as_dictionary(data_file):
    commands_dict = {}
    with open(data_file, "r") as fd:
        new_dataset = fd.readlines()

    for line in new_dataset:
        command, action_dict = line.strip().split("|")
        commands_dict[command] = action_dict

    return commands_dict


def update_file(file_to_update, updated_data):
    """For each command in the file_to_update, apply the updated data if any."""
    with open(file_to_update, "r") as fd:
        old_dataset = fd.readlines()

    commands_dict = load_file_as_dictionary(updated_data)
    updated_new_commands = []

    for line in old_dataset:
        command, action_dict = line.strip().split("|")
        if command in commands_dict:
            updated_new_commands.append("{}|{}".format(command, commands_dict[command]))
        else:
            updated_new_commands.append("{}|{}".format(command, action_dict))

    with open(file_to_update, "w") as fd:
        for line in updated_new_commands:
            fd.write(str(line) + "\n")


if __name__ == "__main__":
    print("========== Preparing Datasets for Model Training ==========")
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_data_dir", type=str, help="craftassist/agent/datasets/full_data/")
    parser.add_argument("--commands_path", type=str, help="path to commands from current split")
    parser.add_argument(
        "--file_to_update",
        type=str,
        default="craftassist/agent/datasets/full_data/annotated_old.txt",
        help="path to file in current data split, eg. train/annotated.txt. \
            This is also the file that will have updated parse trees applied.",
    )
    parser.add_argument(
        "--updated_data",
        type=str,
        default="craftassist/agent/datasets/full_data/annotated.txt",
        help="path to file containing updated parse trees, i.e. in craftassist/agent/full_data",
    )

    args = parser.parse_args()
    update_file(args.file_to_update, args.updated_data)
