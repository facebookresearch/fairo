"""
This file updates the data format as of 05/06 to move repeat_key : "ALL"
to "return_quantity" in selectors. Issue: https://github.com/facebookresearch/fairo/issues/219
changeset: https://www.internalfb.com/phabricator/paste/view/P413307559
"""
import argparse
import json
import copy
from pprint import pprint
from os import walk


def check_repeat_all(d):
    for key, value in d.items():
        if key == "repeat_key" and value == "ALL":
            return True
        elif type(value) == dict:
            if check_repeat_all(value):
                return True
    return False


# Handle various places where repeat all could have possibly occured
def update_repeat_all(d):
    new_d = copy.deepcopy(d)
    for key, value in d.items():
        if (
            key == "reference_object"
            and "filters" in value
            and "repeat" in value["filters"]
            and "repeat_key" in value["filters"]["repeat"]
            and value["filters"]["repeat"]["repeat_key"] == "ALL"
        ):
            if "selector" not in new_d[key]["filters"]:
                new_d[key]["filters"]["selector"] = {}
            if "return_quantity" not in new_d[key]["filters"]["selector"]:
                new_d[key]["filters"]["selector"]["return_quantity"] = {}
            new_d[key]["filters"]["selector"]["return_quantity"] = "ALL"
            new_d[key]["filters"].pop("repeat")
        elif (
            key == "schematic"
            and "filters" in value
            and "repeat" in value["filters"]
            and "repeat_key" in value["filters"]["repeat"]
            and value["filters"]["repeat"]["repeat_key"] == "ALL"
        ):
            if "selector" not in new_d[key]["filters"]:
                new_d[key]["filters"]["selector"] = {}
            if "return_quantity" not in new_d[key]["filters"]["selector"]:
                new_d[key]["filters"]["selector"]["return_quantity"] = {}
            new_d[key]["filters"]["selector"]["return_quantity"] = "ALL"
            new_d[key]["filters"].pop("repeat")
        elif (
            key == "schematic"
            and "repeat" in value
            and "repeat_key" in value["repeat"]
            and value["repeat"]["repeat_key"] == "ALL"
        ):
            if "filters" not in new_d[key]:
                new_d[key]["filters"] = {}
            if "selector" not in new_d[key]["filters"]:
                new_d[key]["filters"]["selector"] = {}
            if "return_quantity" not in new_d[key]["filters"]["selector"]:
                new_d[key]["filters"]["selector"]["return_quantity"] = {}
            new_d[key]["filters"]["selector"]["return_quantity"] = "ALL"
            new_d[key].pop("repeat")
        elif (
            key == "reference_object"
            and "repeat" in value
            and "repeat_key" in value["repeat"]
            and value["repeat"]["repeat_key"] == "ALL"
        ):
            if "filters" not in new_d[key]:
                new_d[key]["filters"] = {}
            if "selector" not in new_d[key]["filters"]:
                new_d[key]["filters"]["selector"] = {}
            if "return_quantity" not in new_d[key]["filters"]["selector"]:
                new_d[key]["filters"]["selector"]["return_quantity"] = {}
            new_d[key]["filters"]["selector"]["return_quantity"] = "ALL"
            new_d[key].pop("repeat")
        elif (
            key == "filters"
            and "repeat" in value
            and "repeat_key" in value["repeat"]
            and value["repeat"]["repeat_key"] == "ALL"
        ):
            if "selector" not in new_d[key]:
                new_d[key]["selector"] = {}
            if "return_quantity" not in new_d[key]["selector"]:
                new_d[key]["selector"]["return_quantity"] = {}
            new_d[key]["selector"]["return_quantity"] = "ALL"
            new_d[key].pop("repeat")
        elif type(value) == dict:
            new_d[key] = update_repeat_all(value)

    return new_d


def update_data(folder):
    """This function walks through the folder and for each file,
    performs update on the dataset and writes output to a new
    file called : f_name + "_new.txt" (templated.txt -> templated_new.txt)
    """
    f = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f_name in filenames:
            count = 0
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
                            if check_repeat_all(action):
                                flag = True
                                action_type = action["action_type"]
                                if action_type in ["UNDO", "RESUME"] and "repeat" in action:
                                    action.pop("repeat")
                                    updated_dict = action
                                else:
                                    if "repeat" in action:
                                        # if repeat is in action, move to child first.
                                        if "reference_object" in action:
                                            action["reference_object"]["repeat"] = action["repeat"]
                                            action.pop("repeat")
                                        elif "schematic" in action:
                                            action["schematic"]["repeat"] = action["repeat"]
                                            action.pop("repeat")
                                        elif "location" in action:
                                            if "reference_object" in action["location"]:
                                                action["location"]["reference_object"][
                                                    "repeat"
                                                ] = action["repeat"]
                                            action.pop("repeat")
                                    updated_dict = update_repeat_all(action)
                                new_actions.append(updated_dict)
                            else:
                                new_actions.append(action)
                        for key in all_keys:
                            new_ad[key] = action_dict[key]
                        new_ad["action_sequence"] = new_actions
                        new_data.append([chat, new_ad])
                    else:
                        if check_repeat_all(action_dict):
                            new_ad = update_repeat_all(action_dict)
                            flag = True
                            new_data.append([chat, new_ad])
                        else:
                            new_data.append([chat, action_dict])
                    if flag:
                        count += 1
                        if count < 10:
                            print(chat)
                            print("BEFORE...")
                            pprint(action_dict)
                            print("AFTER...")
                            pprint(new_ad)
                            print("*" * 80)
            print("filename: %r" % (f_name))
            print("Total updates made in this file : %r" % (count))
            print("*" * 40)

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
                            if check_repeat_all(action):
                                flag = True
                    else:
                        if check_repeat_all(action_dict):
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
