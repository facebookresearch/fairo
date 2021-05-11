"""
This file updates the data format as of 03/24 to add the filters to
schematics change
proposed in : https://github.com/facebookresearch/droidlet/pull/276
issue: https://github.com/facebookresearch/droidlet/issues/219
changeset: https://www.internalfb.com/phabricator/paste/view/P359634791
"""
import argparse
import json
import copy
from pprint import pprint
from os import walk


def update_schematic(folder):
    """This function walks through the folder and for each file,
    performs update on the Dig and Build actions in dataset and
    writes output to a new file called : f_name + "_new.txt"
    (templated.txt -> templated_new.txt)
    """
    f = []
    action_list = ["DIG", "BUILD"]
    for (dirpath, dirnames, filenames) in walk(folder):
        for f_name in filenames:
            count = 0
            count1 = 0
            if f_name == "templated_modify.txt":
                continue
            print("processing input file : %r" % (f_name))
            file = folder + f_name
            with open(file) as f:
                new_data = []
                count = 0
                for line in f.readlines():
                    chat, action_dict = line.strip().split("|")
                    action_dict = json.loads(action_dict)
                    if action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND":
                        all_keys = list(action_dict.keys())
                        all_keys.remove("action_sequence")
                        actions = action_dict["action_sequence"]
                        for i, action in enumerate(actions):
                            if (action["action_type"] in actio) and i > 0:
                                if "repeat" in action and "repeat_count" in action["repeat"]:
                                    print(chat)
                                    print(action_dict)
                                    count1 += 1
                        # for action in actions: ->> only for annotated.
                        action = actions[0]
                        if action["action_type"] in action_list:
                            if action["action_type"] == "BUILD" and "reference_object" in action:
                                # it is a copy
                                new_data.append([chat, action_dict])
                                continue
                            # FOR loop
                            if "repeat" in action and "repeat_count" in action["repeat"]:
                                count_span = action["repeat"]["repeat_count"]
                                action.pop("repeat")
                                if "schematic" not in action:
                                    action["schematic"] = {}
                                if "filters" not in action["schematic"]:
                                    action["schematic"]["filters"] = {}
                                if "selector" not in action["schematic"]["filters"]:
                                    action["schematic"]["filters"]["selector"] = {}
                                if (
                                    "return_quantity"
                                    not in action["schematic"]["filters"]["selector"]
                                ):
                                    action["schematic"]["filters"]["selector"][
                                        "return_quantity"
                                    ] = {}
                                action["schematic"]["filters"]["selector"]["return_quantity"][
                                    "random"
                                ] = count_span
                                action["schematic"]["filters"]["selector"]["same"] = "ALLOWED"
                                count += 1
                    new_data.append([chat, action_dict])

            out_file_name = f_name.split(".txt")[0] + "_new.txt"
            out_file = folder + out_file_name
            print("Writing to output file: %r" % (out_file))
            with open(out_file, "w") as f:
                for item in new_data:
                    chat, action_dict = item
                    f.write(chat + "|" + json.dumps(action_dict) + "\n")
            print("*" * 20)


def update_reference_object(folder):
    """This function walks through the folder and for each file,
    performs update on few actions in dataset and
    writes output to a new file called : f_name + "_new.txt"
    (templated.txt -> templated_new.txt)
    """
    f = []
    action_list = ["DESTROY", "SPAWN", "BUILD", "FILL", "GET", "SCOUT", "OTHERACTION"]
    for (dirpath, dirnames, filenames) in walk(folder):
        for f_name in filenames:
            count = 0
            count1 = 0
            if f_name == "templated_modify.txt":
                continue
            print("processing input file : %r" % (f_name))
            file = folder + f_name
            with open(file) as f:
                new_data = []
                count = 0
                for line in f.readlines():
                    chat, action_dict = line.strip().split("|")
                    action_dict = json.loads(action_dict)
                    if action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND":
                        all_keys = list(action_dict.keys())
                        all_keys.remove("action_sequence")
                        actions = action_dict["action_sequence"]
                        for i, action in enumerate(actions):
                            if (action["action_type"] in action_list) and i > 0:
                                if "repeat" in action and "repeat_count" in action["repeat"]:
                                    count1 += 1
                            if action["action_type"] in action_list:
                                if action["action_type"] == "BUILD" and "schematic" in action:
                                    # it is a build
                                    continue
                                # FOR loop
                                if "repeat" in action and "repeat_count" in action["repeat"]:
                                    count_span = action["repeat"]["repeat_count"]
                                    action.pop("repeat")
                                    if "reference_object" not in action:
                                        action["reference_object"] = {}
                                    if "filters" not in action["reference_object"]:
                                        action["reference_object"]["filters"] = {}
                                    if "selector" not in action["reference_object"]["filters"]:
                                        action["reference_object"]["filters"]["selector"] = {}
                                    if (
                                        "random_quantity"
                                        not in action["reference_object"]["filters"]["selector"]
                                    ):
                                        action["reference_object"]["filters"]["selector"][
                                            "random_quantity"
                                        ] = {}
                                    action["reference_object"]["filters"]["selector"][
                                        "random_quantity"
                                    ]["random"] = count_span
                                    action["reference_object"]["filters"]["selector"][
                                        "same"
                                    ] = "DISALLOWED"
                                    count += 1
                    new_data.append([chat, action_dict])

            print(count)
            print(count1)
            print(len(new_data))
            print("*" * 40)

            out_file_name = f_name.split(".txt")[0] + "_new.txt"
            out_file = folder + out_file_name
            print("Writing to output file: %r" % (out_file))
            with open(out_file, "w") as f:
                for item in new_data:
                    chat, action_dict = item
                    f.write(chat + "|" + json.dumps(action_dict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        help="The folder containing all files that need to be updated",
        # Assuming run from ~/droidlet
        default="craftassist/agent/datasets/full_data/",
    )
    args = parser.parse_args()
    update_schematic(args.input_folder)
    update_reference_object(args.input_folder)
