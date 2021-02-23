import glob
from random import shuffle
import re
import argparse

"""
Create static validation set
2% of each data type
"""

FULL_DATA_DIR = "craftassist/agent/datasets/full_data/"
DATA_CHECKPOINT = "/checkpoint/rebeccaqian/datasets/01_28/static_valid/"
COMMANDS_CHECKPOINT = "/checkpoint/rebeccaqian/datasets/01_07/static_valid/commands/"
TRAIN_VALID_DATA_CHECKPOINT = "/checkpoint/rebeccaqian/datasets/01_28/train_valid/"

def write_data_chunk_to_file(data, file, commands_only=False):
    with open(file, "w") as fd:
        for line in data:
            if commands_only:
                command, action_dict = line.split("|")
                fd.write(command + "\n")
            else:
                fd.write(line)


def update_data_chunk(updated_data, commands_path, output_dir):
    """Read the commands in file_to_update and update the commands with the updated trees
    in updated_data.
    """
    VALID_UPDATED_SET = []
    TRAIN_UPDATED_SET = []

    with open(updated_data, "r") as fd:
        dataset = fd.readlines()
    for partition in ["train", "valid", "test"]:
        chunk_commands = [r.strip() for r in open("{}/{}/{}"commands_).readlines()]

    for row in dataset:
        command, action_dict = row.split("|")
        if command in chunk_commands:
            VALID_UPDATED_SET.append("{}|{}".format(command, action_dict))
        else:
            TRAIN_UPDATED_SET.append("{}|{}".format(command, action_dict))
    print(len(TRAIN_UPDATED_SET))
    # print(file_to_update)
    # Write the updated chunk
    write_data_chunk_to_file(TRAIN_UPDATED_SET, output_dir + "../train/")
    write_data_chunk_to_file(VALID_UPDATED_SET, output_dir + "../valid/")


def update_data_dir(full_data_dir, commands_path):
    """For each file in the data dir, separate the train and valid sets based on previous split.
    """
    for name in glob.glob("{}/*".format(full_data_dir)):
        print(name)
	data_type = 
        for partition in ["valid", "test"]:
		chunk_commands = [r.strip() for r in open("{}/{}/{}".format(commands_path, partition, data_type)).readlines()]
        update_data_chunk(name, commands_path, full_data_dir)

# for name in glob.glob(FULL_DATA_DIR + "*"): 
#     print(name) 
#     VALID_UPDATED_SET = []
#     TRAIN_UPDATED_SET = []
#         print(len(dataset))
#         # shuffle(dataset)
#         # num_static_samples = round(len(dataset) * 0.02)
#         # # print(num_static_samples)
#         # static_samples = dataset[:num_static_samples]
#         # remaining_samples = dataset[num_static_samples:]
#         # print(len(static_samples))
#         # # Write the full static samples to a file
#         data_type_name = re.search("([^\/]+$)", name).group(1)
#         print(data_type_name)
#         # new_filename = DATA_CHECKPOINT + data_type_name
#         # write_data_chunk_to_file(static_samples, new_filename)
#         # grab the commands
#         # commands_filename = COMMANDS_CHECKPOINT + data_type_name
#         # write_data_chunk_to_file(static_samples, commands_filename, commands_only=True)

        # # Load the static commands
        # # for each command, if its in there, put it iin valid set
        # static_valid_commands = [r.strip() for r in open(COMMANDS_CHECKPOINT + data_type_name).readlines()]
        # # import ipdb; ipdb.set_trace()
        # for row in dataset:
        #     command, action_dict = row.split("|")
        #     command = command #.strip()
        #     if command in static_valid_commands:
        #         VALID_UPDATED_SET.append("{}|{}".format(command, action_dict))
        #     else:
        #         TRAIN_UPDATED_SET.append("{}|{}".format(command, action_dict))
        #         # import ipdb; ipdb.set_trace()
        # # if not, put it in the other set
        # # write both chunks
        # # write the remaining chunks as new data
        # # train_path = TRAIN_VALID_DATA_CHECKPOINT + data_type_name
        # # write_data_chunk_to_file(remaining_samples, train_path)
        # # write the static set out
        # import ipdb; ipdb.set_trace()
        # print(len(TRAIN_UPDATED_SET))
        # print(TRAIN_VALID_DATA_CHECKPOINT)
        # new_filename = DATA_CHECKPOINT + data_type_name
        # write_data_chunk_to_file(VALID_UPDATED_SET, new_filename)
        # # write the train set out
        # train_path = TRAIN_VALID_DATA_CHECKPOINT + data_type_name
        # write_data_chunk_to_file(TRAIN_UPDATED_SET, train_path)
        # # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    print("========== Preparing Datasets for Model Training ==========")
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--full_data_dir",
            type=str,
            help="path to dataset we want to update"
    )
    parser.add_argument(
            "--commands_path",
            type=str,
            help="path to commands from current split"
    )

    args = parser.parse_args()
    full_data_dir = args.full_data_dir
    commands_path = args.commands_path
    print(full_data_dir)
    # For each split in the directory, for each data file, load the commands and update the tree 
    update_data_dir(full_data_dir, commands_path)
