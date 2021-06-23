import ast
import json
import copy
from pprint import pprint
from os import path


# output of tool D
parent_folder = "/Users/rebeccaqian/minecraft/tools/annotation_tools/turk_with_s3/"
folder_name_D = parent_folder + "D/"
tool_D_out_file = folder_name_D + "all_agreements.txt"

# output of tool C
parent_folder = "/Users/rebeccaqian/minecraft/tools/annotation_tools/turk_with_s3/"
folder_name_C = parent_folder + "C/"
tool_C_out_file = folder_name_C + "all_agreements.txt"

# combine outputs
# check if all keys of tool C annotated yes -> put directly
# if no , check child in t2 and combine
# construct mape of tool 1
toolC_map = {}

if path.exists(tool_C_out_file):
    with open(tool_C_out_file) as f:
        for line in f.readlines():
            line = line.strip()
            cmd, ref_obj_text, a_d = line.split("\t")
            if cmd in toolC_map:
                toolC_map[cmd].update(ast.literal_eval(a_d))
            else:
                toolC_map[cmd] = ast.literal_eval(a_d)
print(len(toolC_map.keys()))

# construct map of tool 2
toolD_map = {}

if path.exists(tool_D_out_file):
    with open(tool_D_out_file) as f2:
        for line in f2.readlines():
            line = line.strip()
            cmd, comparison_text, comparison_dict = line.split("\t")
            if cmd in toolD_map:
                print("BUGGGGG")
            # add the comparison dict to command -> dict
            toolD_map[cmd] = ast.literal_eval(comparison_dict)
print(len(toolD_map.keys()))


def all_yes(a_dict):
    if type(a_dict) == str:
        a_dict = ast.literal_eval(a_dict)
    for k, val in a_dict.items():
        if type(val) == list and val[0] == "no":
            return False
    return True


def clean_up_dict(a_dict):
    if type(a_dict) == str:
        a_dict = ast.literal_eval(a_dict)
    new_d = {}
    for k, val in a_dict.items():
        if type(val) == list:
            if val[0] in ["yes", "no"]:
                new_d[k] = val[1]
        elif type(val) == dict:
            new_d[k] = clean_up_dict(val)
        else:
            new_d[k] = val
    return new_d


# post_process spans and "contains_coreference" : "no"
def merge_indices(indices):
    a, b = indices[0]
    for i in range(1, len(indices)):
        a = min(a, indices[i][0])
        b = max(b, indices[i][1])
    return [a, b]


def fix_spans(d):
    new_d = {}
    if type(d) == str:
        d = ast.literal_eval(d)
    for k, v in d.items():
        if k == "contains_coreference" and v == "no":
            continue
        if type(v) == list:
            new_d[k] = [0, merge_indices(v)]
            continue
        elif type(v) == dict:
            new_d[k] = fix_spans(v)
            continue
        else:
            new_d[k] = v
    return new_d


def fix_ref_obj(clean_dict):
    val = clean_dict
    new_clean_dict = {}
    if "special_reference" in val:
        new_clean_dict["special_reference"] = val["special_reference"]
        val.pop("special_reference")
    if "repeat" in val:
        new_clean_dict["repeat"] = val["repeat"]
        val.pop("repeat")
    if val:
        new_clean_dict["filters"] = val
    return new_clean_dict


# combine and write output to a file
i = 0
# what these action will look like in the map

toolC_updated_map = {}

# update dict of toolC with tool D and keep that in tool C's map
for cmd, a_dict in toolC_map.items():
    # remove the ['yes', val] etc
    for key in a_dict.keys():
        a_dict_child = a_dict[key]
        clean_dict = clean_up_dict(a_dict_child)
        # fix reference object inside location of reference object
        if "location" in clean_dict and "reference_object" in clean_dict["location"]:
            value = clean_dict["location"]["reference_object"]
            clean_dict["location"]["reference_object"] = fix_ref_obj(value)
        new_clean_dict = fix_ref_obj(clean_dict)

        if all_yes(a_dict_child):
            if cmd in toolC_updated_map:
                toolC_updated_map[cmd][key] = new_clean_dict
            else:
                toolC_updated_map[cmd] = {key: new_clean_dict}
            continue
        new_clean_dict.pop("comparison", None)
        comparison_dict = toolD_map[cmd]  # check on this again

        valid_dict = {}
        valid_dict[key] = {}
        valid_dict[key]["filters"] = new_clean_dict
        valid_dict[key]["filters"].update(comparison_dict)
        toolC_updated_map[cmd] = valid_dict  # only gets populated if filters exist
pprint(toolC_updated_map)


print(len(toolC_updated_map.keys()))
print(len(toolC_map.keys()))

# output of tool 1
folder_name_A = parent_folder + "A/"
tool_A_out_file = folder_name_A + "all_agreements.txt"

# output of tool 2
folder_name_B = parent_folder + "B/"
tool_B_out_file = folder_name_B + "all_agreements.txt"

# combine outputs
# check if all keys of t1 annotated yes -> put directly
# if no , check child in t2 and combine
# construct mape of tool 1
toolA_map = {}
with open(tool_A_out_file) as f:
    for line in f.readlines():
        line = line.strip()
        cmd, a_d = line.split("\t")
        toolA_map[cmd] = a_d
print(len(toolA_map.keys()))

# construct map of tool 2
toolB_map = {}

if path.isfile(tool_B_out_file):
    with open(tool_B_out_file) as f2:
        for line in f2.readlines():
            line = line.strip()
            cmd, child, child_dict = line.split("\t")
            if cmd in toolB_map and child in toolB_map[cmd]:
                print("BUGGG")
            if cmd not in toolB_map:
                toolB_map[cmd] = {}
            toolB_map[cmd][child] = child_dict
print(len(toolB_map.keys()))


def all_yes(a_dict):
    if type(a_dict) == str:
        a_dict = ast.literal_eval(a_dict)
    for k, val in a_dict.items():
        if type(val) == list and val[0] == "no":
            return False
    return True


def clean_dict_1(a_dict):
    if type(a_dict) == str:
        a_dict = ast.literal_eval(a_dict)
    new_d = {}
    for k, val in a_dict.items():
        if type(val) == list:
            if val[0] in ["yes", "no"]:
                new_d[k] = val[1]
        elif type(val) == dict:
            new_d[k] = a_dict(val[1])
        else:
            new_d[k] = val
    # only for now
    if "dance_type_span" in new_d:
        new_d["dance_type"] = {}
        new_d["dance_type"]["dance_type_name"] = new_d["dance_type_span"]
        new_d.pop("dance_type_span")
    if "dance_type_name" in new_d:
        new_d["dance_type"] = {}
        new_d["dance_type"]["dance_type_name"] = new_d["dance_type_name"]
        new_d.pop("dance_type_name")
    return new_d


# post_process spans and "contains_coreference" : "no"
def merge_indices(indices):
    a, b = indices[0]
    for i in range(1, len(indices)):
        a = min(a, indices[i][0])
        b = max(b, indices[i][1])
    return [a, b]


def fix_put_mem(d):

    if type(d) == str:
        d = ast.literal_eval(d)
    new_d = copy.deepcopy(d)
    del new_d["action_type"]
    if "has_tag" in new_d and "upsert" in new_d:
        new_d["upsert"]["memory_data"]["has_tag"] = new_d["has_tag"]
        del new_d["has_tag"]

    return new_d


def fix_spans(d):
    new_d = {}
    if type(d) == str:
        d = ast.literal_eval(d)
    for k, v in d.items():
        if k == "contains_coreference" and v == "no":
            continue
        if type(v) == list:
            if k == "tag_val":
                new_d["has_tag"] = [0, merge_indices(v)]
            else:
                new_d[k] = [0, merge_indices(v)]
            continue
        elif type(v) == dict:
            new_d[k] = fix_spans(v)
            continue
        else:
            new_d[k] = v
    return new_d


# combine and write output to a file
i = 0
# what these action will look like in the map
dance_type_map = {"point": "point", "look": "look_turn", "turn": "body_turn"}

# update dict of tool1 with tool 2
with open(parent_folder + "/all_combined.txt", "w") as f:
    for cmd, a_dict in toolA_map.items():
        # remove the ['yes', val] etc
        clean_dict = clean_dict_1(a_dict)
        print(clean_dict)
        if all_yes(a_dict):
            action_type = clean_dict["action_type"]

            valid_dict = {}
            valid_dict["dialogue_type"] = clean_dict["dialogue_type"]
            del clean_dict["dialogue_type"]
            clean_dict["action_type"] = clean_dict["action_type"].upper()
            act_dict = fix_spans(clean_dict)
            valid_dict["action_sequence"] = [act_dict]

            f.write(cmd + "\t" + json.dumps(valid_dict) + "\n")
            print(cmd)
            print(valid_dict)
            print("All yes")
            print("*" * 20)
            continue
        if clean_dict["action_type"] == "noop":
            f.write(cmd + "\t" + json.dumps(clean_dict) + "\n")
            print(clean_dict)
            print("NOOP")
            print("*" * 20)
            continue
        if clean_dict["action_type"] == "otheraction":
            f.write(cmd + "\t" + str(a_dict) + "\n")
            continue

        if toolB_map and cmd in toolB_map:
            child_dict_all = toolB_map[cmd]
            # update action dict with all children except for reference object
            for k, v in child_dict_all.items():
                if k not in clean_dict:
                    print("BUGGGG")
                if type(v) == str:
                    v = ast.literal_eval(v)
                if not v:
                    continue

                if "reference_object" in v[k]:
                    value = v[k]["reference_object"]
                    v[k]["reference_object"] = fix_ref_obj(value)
                if k == "tag_val":
                    clean_dict.update(v)
                elif k == "facing":
                    action_type = clean_dict["action_type"]
                    # set to dance
                    clean_dict["action_type"] = "DANCE"
                    clean_dict["dance_type"] = {dance_type_map[action_type]: v["facing"]}
                    clean_dict.pop("facing")
                else:
                    clean_dict[k] = v[k]
        ref_obj_dict = {}
        if toolC_updated_map and cmd in toolC_updated_map:
            ref_obj_dict = toolC_updated_map[cmd]
        clean_dict.update(ref_obj_dict)
        if "receiver_reference_object" in clean_dict:
            clean_dict["receiver"] = {"reference_object": clean_dict["receiver_reference_object"]}
            clean_dict.pop("receiver_reference_object")
        if "receiver_location" in clean_dict:
            clean_dict["receiver"] = {"location": clean_dict["receiver_location"]}
            clean_dict.pop("receiver_location")

        actual_dict = copy.deepcopy((clean_dict))

        action_type = actual_dict["action_type"]

        valid_dict = {}
        valid_dict["dialogue_type"] = actual_dict["dialogue_type"]
        del actual_dict["dialogue_type"]
        actual_dict["action_type"] = actual_dict["action_type"].upper()
        act_dict = fix_spans(actual_dict)
        valid_dict["action_sequence"] = [act_dict]
        print(cmd)
        pprint(valid_dict)
        print("*" * 40)
        f.write(cmd + "\t" + json.dumps(valid_dict) + "\n")
