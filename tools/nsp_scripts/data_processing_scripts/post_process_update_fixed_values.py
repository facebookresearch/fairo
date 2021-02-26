"""
This file updates the data format as of 02/24 to add fixed values change
proposed in : https://github.com/facebookresearch/droidlet/pull/190
issue: https://github.com/facebookresearch/droidlet/issues/191
Examples of change : https://www.internalfb.com/phabricator/paste/view/P217893342 
"""
import argparse
import json
import copy

from os import walk


def check_fixed_value(d):
    for key, value in d.items():
        if key in [
            "author",
            "obj_text",
            "frame",
            "ordinal",
            "close_tolerance",
            "modulus",
            "special_reference",
        ]:
            val = d[key]
            if type(val) == str:
                return True
        elif key in ["relative_yaw", "relative_pitch"]:
            if type(value) == dict and "angle" in value and type(value["angle"] == str):
                return True
        elif key == "triples":
            for entry in value:
                if check_fixed_value(entry):
                    return True
        elif type(value) == dict:
            if check_fixed_value(value):
                return True
    return False


def updated_fixed_value(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if key in [
            "author",
            "obj_text",
            "frame",
            "ordinal",
            "close_tolerance",
            "modulus",
            "special_reference",
        ]:
            val = d[key]
            if type(val) == str:
                if val.startswith("_"):
                    val = val[1:]
                val = val.upper()
                new_d[key] = {"fixed_value": val}
        elif key in ["relative_yaw", "relative_pitch"]:
            if type(value) == dict and "angle" in value:
                new_d[key] = {"fixed_value": value["angle"]}
        elif key == "triples":
            new_value = []
            for entry in value:
                new_val = updated_fixed_value(entry)
                new_value.append(new_val)
            new_d[key] = new_value
        elif type(value) == dict:
            new_d[key] = updated_fixed_value(value)
    return new_d


def update_data(folder):
    """This function walks through the folder and for each file,
    performs update on the dataset and writes output to a new
    file called : f_name + "_new.txt" (templated.txt -> templated_new.txt)
    """
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
                            if check_fixed_value(action):
                                flag = True
                                updated_dict = updated_fixed_value(action)
                                new_actions.append(updated_dict)
                            else:
                                new_actions.append(action)

                        for key in all_keys:
                            new_ad[key] = action_dict[key]
                        new_ad["action_sequence"] = new_actions
                        new_data.append([chat, new_ad])
                    else:
                        if check_fixed_value(action_dict):
                            flag = True
                            new_ad = updated_fixed_value(action_dict)
                            new_data.append([chat, new_ad])
                        else:
                            new_data.append([chat, action_dict])
                    if flag:
                        flag = False
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
                            if check_fixed_value(action):
                                flag = True
                    else:
                        if check_fixed_value(action_dict):
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
        default="craftassist/agent/datasets/full_data/",
    )
    args = parser.parse_args()
    update_data(args.input_folder)

