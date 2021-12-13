"""
This file updates the data format as of 03/12 to add the filters to
schematics change
proposed in : https://github.com/facebookresearch/fairo/pull/253
issue: https://github.com/facebookresearch/fairo/issues/219
"""
import json
import copy
import argparse
from os import walk
from pprint import pprint


def check_dance_type(d):
    for key, value in d.items():
        if key == "dance_type" and (
            ("dance_type_tag" in value)
            or ("dance_type_name" in value)
            or ("dance_type_span" in value)
        ):
            return True
        elif type(value) == dict:
            if check_dance_type(value):
                return True
    return False


def update_dance_type(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if key == "dance_type":
            val = d[key]
            if type(val) == dict:
                if "dance_type_name" in val:
                    if "filters" not in new_d[key]:
                        new_d[key]["filters"] = {}
                    if "triples" not in new_d[key]["filters"]:
                        new_d[key]["filters"]["triples"] = []
                    new_d[key]["filters"]["triples"].append(
                        {"pred_text": "has_name", "obj_text": val["dance_type_name"]}
                    )
                    new_d[key].pop("dance_type_name")
                if "dance_type_span" in val:
                    if "filters" not in new_d[key]:
                        new_d[key]["filters"] = {}
                    if "triples" not in new_d[key]["filters"]:
                        new_d[key]["filters"]["triples"] = []
                    new_d[key]["filters"]["triples"].append(
                        {"pred_text": "has_tag", "obj_text": val["dance_type_span"]}
                    )
                    new_d[key].pop("dance_type_span")
                if "dance_type_tag" in val:
                    if "filters" not in new_d[key]:
                        new_d[key]["filters"] = {}
                    if "triples" not in new_d[key]["filters"]:
                        new_d[key]["filters"]["triples"] = []
                    new_d[key]["filters"]["triples"].append(
                        {"pred_text": "has_tag", "obj_text": val["dance_type_tag"]}
                    )
                    new_d[key].pop("dance_type_tag")
        elif type(value) == dict:
            new_d[key] = update_dance_type(value)

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
                            if check_dance_type(action):
                                flag = True
                                updated_dict = update_dance_type(action)
                                new_actions.append(updated_dict)
                            else:
                                new_actions.append(action)

                        for key in all_keys:
                            new_ad[key] = action_dict[key]
                        new_ad["action_sequence"] = new_actions
                        new_data.append([chat, new_ad])
                    else:
                        if check_dance_type(action_dict):
                            new_ad = update_dance_type(action_dict)
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
                            if check_dance_type(action):
                                flag = True
                    else:
                        if check_dance_type(action_dict):
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
