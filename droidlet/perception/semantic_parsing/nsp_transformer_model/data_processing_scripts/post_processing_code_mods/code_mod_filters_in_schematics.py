"""
This file updates the data format as of 03/10 to add the filters to
schematics change
proposed in : https://github.com/facebookresearch/fairo/pull/236
issue: https://github.com/facebookresearch/fairo/issues/219
"""
import json
import copy
import argparse
from os import walk
from pprint import pprint


def check_schematic(d):
    for key, value in d.items():
        if key == "schematic":
            return True
        elif type(value) == dict:
            if check_schematic(value):
                return True
    return False


def check_schematics_triples(d):
    for key, val in d.items():
        if key == "schematic" and type(val) == dict and "triples" in val:
            return True
        elif type(val) == dict:
            if check_schematics_triples(val):
                return True
    return False


def update_schematics(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if key == "schematic":
            val = d[key]
            if type(val) == dict and "triples" in val:
                new_d[key]["filters"] = {"triples": val["triples"]}
                new_d[key].pop("triples")
        elif type(value) == dict:
            new_d[key] = update_schematics(value)

    return new_d


def update_data(folder):
    """This function walks through the folder and for each file,
    performs update on the dataset and writes output to a new
    file called : f_name + "_new.txt" (templated.txt -> templated_new.txt)
    """
    f = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f_name in filenames:
            if f_name == "templated_modify.txt":
                continue
            print("processing input file : %r" % (f_name))
            file = folder + f_name
            with open(file) as f:
                new_data = []
                count = 0
                for line in f.readlines():
                    flag = False
                    chat, action_dict = line.strip().split("|")
                    action_dict = json.loads(action_dict)
                    if action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND":
                        all_keys = list(action_dict.keys())
                        all_keys.remove("action_sequence")
                        actions = action_dict["action_sequence"]
                        new_actions = []
                        new_ad = {}
                        for action in actions:
                            if check_schematic(action):
                                flag = True
                                updated_dict = update_schematics(action)
                                new_actions.append(updated_dict)
                            else:
                                new_actions.append(action)

                        for key in all_keys:
                            new_ad[key] = action_dict[key]
                        new_ad["action_sequence"] = new_actions
                        new_data.append([chat, new_ad])
                    else:
                        if check_schematic(action_dict):
                            new_ad = update_schematics(action_dict)
                            flag = True
                            new_data.append([chat, new_ad])
                        else:
                            new_data.append([chat, action_dict])
                    if flag:
                        flag = False
                        print(chat)
                        print("before...")
                        pprint(action_dict)
                        print("after...")
                        pprint(new_ad)
                        print("*" * 20)
                        count += 1
            print("Total number of updates done: %r" % (count))

            out_file_name = f_name.split(".txt")[0] + "_new.txt"
            out_file = folder + out_file_name
            print("Writing to output file: %r" % (out_file))
            with open(out_file, "w") as f:
                for item in new_data:
                    chat, action_dict = item
                    f.write(chat + "|" + json.dumps(action_dict) + "\n")

            print("Now computing num updates on output file...")
            count = 0
            with open(out_file) as f:
                for line in f.readlines():
                    flag = False
                    chat, action_dict = line.strip().split("|")
                    action_dict = json.loads(action_dict)
                    if action_dict["dialogue_type"] == "HUMAN_GIVE_COMMAND":
                        actions = action_dict["action_sequence"]
                        for action in actions:
                            if check_schematics_triples(action):
                                flag = True
                    else:
                        if check_schematics_triples(action_dict):
                            flag = True
                    if flag:
                        count += 1
            print("Total number of updates needed in out file: %r" % (count))
            print("*" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        help="The folder containing all files that need to be updated",
        # Assuming run from ~/droidlet
        default="droidlet/artifacts/datasets/full_data/",
    )
    args = parser.parse_args()
    update_data(args.input_folder)
