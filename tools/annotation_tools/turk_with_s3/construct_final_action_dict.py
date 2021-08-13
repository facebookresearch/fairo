import ast
import json
import copy
from pprint import pprint
import os
import argparse

# Construct maps of tools A->D
toolC_map = {}
toolD_map = {}
toolA_map = {}
toolB_map = {}
toolC_updated_map = {}


def collect_tool_outputs(tool_C_out_file, tool_D_out_file):
    # check if all keys of tool C annotated yes -> put directly
    # if no , check child in t2 and combine
    if os.path.exists(tool_C_out_file):
        with open(tool_C_out_file) as f:
            for line in f.readlines():
                line = line.strip()
                cmd, ref_obj_text, a_d = line.split("\t")
                if cmd in toolC_map:
                    toolC_map[cmd].update(ast.literal_eval(a_d))
                else:
                    toolC_map[cmd] = ast.literal_eval(a_d)
    # print("toolC map keys")
    # print(toolC_map.keys())

    if os.path.exists(tool_D_out_file):
        with open(tool_D_out_file) as f2:
            for line in f2.readlines():
                line = line.strip()
                cmd, comparison_text, comparison_dict = line.split("\t")
                if cmd in toolD_map:
                    print("Error: command {} is in the tool D outputs".format(cmd))
                # add the comparison dict to command -> dict
                toolD_map[cmd] = ast.literal_eval(comparison_dict)
    # print("toolD map keys")
    # print(toolD_map.keys())


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
            else:
                new_d[k] = val
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

def contiguous_spans(indices):
    a, b = indices[0]
    for i in range(1, len(indices)):
        a1, b1 = indices[i]
        if a1 != b +1 :
            return False
        a, b = a1, b1
    return True

def fix_ref_obj(clean_dict):
    val = clean_dict
    new_clean_dict = {}
    if "special_reference" in val:
        new_clean_dict["special_reference"] = val["special_reference"]
        val.pop("special_reference")
    if val:
        # Add selectors to filters if there is a location
        '''
        "selector": {
        "return_quantity": <ARGVAL> / "RANDOM" / "ALL",
        "ordinal": {"fixed_value" : "FIRST"} / <span>, 
        "location": <LOCATION>,
        "same":"ALLOWED"/"DISALLOWED"/"REQUIRED"
        },
        '''
        if "location" in val:
            if "selector" not in val:
                val["selector"] = {}
            val["selector"]["location"] = val["location"]
            del val["location"]
        # Put has_x attributes in triples
        # Put triples inside "where_clause"
        triples = []
        for k, v in [x for x in val.items()]:
            if "has_" in k:
                triples.append({
                    "pred_text": k,
                    "obj_text": v
                })
                del val[k]
        '''
        "where_clause" : {
        "AND": [<COMPARATOR>/<TRIPLES>], 
        "OR": [<COMPARATOR>/<TRIPLES>], 
        "NOT": [<COMPARATOR>/<TRIPLES>]
        }
        '''
        if len(triples) > 0:
            if "where_clause" not in val:
                val["where_clause"] = {}
            val["where_clause"]["AND"] =  triples
            # val["triples"] = triples
        new_clean_dict["filters"] = val
    return new_clean_dict


def combine_tool_cd_make_ab(tool_A_out_file, tool_B_out_file):
    # combine and write output to a file
    # what these action will look like in the map
    i = 0
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
            new_clean_dict['filters'].pop("comparison", None)
            comparison_dict = toolD_map[cmd]  # check on this again

            valid_dict = {}
            valid_dict[key] = {}
            valid_dict[key].update(new_clean_dict)
            valid_dict[key]["filters"].update(comparison_dict)
            toolC_updated_map[cmd] = valid_dict  # only gets populated if filters exist
    # print("in combine_tool_cd_make_ab...")
    # pprint(toolC_updated_map)

    # combine outputs
    # check if all keys of t1 annotated yes -> put directly
    # if no , check child in t2 and combine
    # construct mape of tool 1
    with open(tool_A_out_file) as f:
        for line in f.readlines():
            line = line.strip()
            cmd, a_d = line.split("\t")
            cmd = cmd.strip()
            toolA_map[cmd] = a_d
    # pprint(toolA_map)
    # construct map of tool 2

    if os.path.isfile(tool_B_out_file):
        with open(tool_B_out_file) as f2:
            for line in f2.readlines():
                line = line.strip()
                cmd, child, child_dict = line.split("\t")
                cmd = cmd.strip()
                child = child.strip()
                if cmd in toolB_map and child in toolB_map[cmd]:
                    print("BUGGG")
                if cmd not in toolB_map:
                    toolB_map[cmd] = {}
                toolB_map[cmd][child] = child_dict
    # pprint(toolB_map)


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
    memory_type = "TRIPLES"
    if "dialogue_target" in new_d:
        if new_d["dialogue_target"] == "f1":
            memory_type = "SET"
            if "selector" in new_d["filters"]:
                new_d["filters"]["selector"]["location"] = {"location_type": "SPEAKER_LOOK"}
                new_d["filters"]["selector"]["same"] = "DISALLOWED"
        elif new_d["dialogue_target"] == "SWARM":
            memory_type = "SET"
            new_d["filters"] = {
                "selector": {
                    "return_quantity": "ALL",
                    "same": "DISALLOWED"
                },
                "memory_type": "AGENT"}
        elif new_d["dialogue_target"] == "AGENT":
            new_d["filters"] = {
                "memory_type": "AGENT",
                "selector" : {"location" : "SPEAKER_LOOK"}
                }
        del new_d['dialogue_target']

    if "has_tag" in new_d:
        new_d["upsert"] = {"memory_data" :{"memory_type" : memory_type}}
        new_d["upsert"]["memory_data"]["triples"] = [{"pred_text": "has_tag", "obj_text": new_d["has_tag"]}]
        del new_d["has_tag"]
    '''
    {'action_type': ['yes', 'tag'],
     'dialogue_target': ['yes', 'f1'],
     'dialogue_type': ['yes', 'PUT_MEMORY'],
     'filters': ['yes', {'selector': {'return_quantity': [[1, 1]]}}],
     'tag_val': ['no', [[4, 4], [5, 5]]]}
    '''

    return new_d


def fix_spans(d):
    new_d = {}
    if type(d) == str:
        d = ast.literal_eval(d)
    for k, v in d.items():
        if k == "contains_coreference" and v == "no":
            continue
        if type(v) == list:
            if k not in ["triples", "AND", "OR", "NOT"]:
                if k == "tag_val":
                    new_d["has_tag"] = [0, merge_indices(v)]
                else:
                    new_d[k] = [0, merge_indices(v)]
            else:
                new_d[k] = [fix_spans(x) for x in v]
                continue
        elif type(v) == dict:
            new_d[k] = fix_spans(v)
            continue
        else:
            new_d[k] = v
    return new_d


def update_action_dictionaries(all_combined_path):
    # combine and write output to a file
    i = 0
    # what these action will look like in the map
    dance_type_map = {"point": "point",
                      "look": "look_turn",
                      "turn": "body_turn"}

    # update dict of tool1 with tool 2
    with open(all_combined_path, "w") as f:
        for cmd, a_dict in toolA_map.items():
            # remove the ['yes', val] etc
            clean_dict = clean_dict_1(a_dict)
            # if cmd!="i will call you alpha":
            #     continue
            # TODO: check repeats here for action level repeat
            if all_yes(a_dict):
                action_type = clean_dict["action_type"]
                valid_dict = {}
                valid_dict["dialogue_type"] = clean_dict["dialogue_type"]
                del clean_dict["dialogue_type"]
                clean_dict["action_type"] = clean_dict["action_type"].upper()
                if "dialogue_target" in clean_dict and clean_dict["dialogue_target"] == "f1":
                    filters_sub_dict = clean_dict["filters"]
                    if "where_clause" in filters_sub_dict:
                        if len(filters_sub_dict["where_clause"]) > 1:
                            # OR
                            triples = []
                            for item in filters_sub_dict["where_clause"]:
                                triples.append({"pred_text": "has_name", "obj_text": [item]})
                            clean_dict["filters"]["where_clause"] = {"OR": triples}
                        elif len(filters_sub_dict["where_clause"]) == 1:
                            # AND
                            clean_dict["filters"]["where_clause"] = {
                                "OR": [filters_sub_dict["where_clause"][0]]
                            }
                act_dict = fix_spans(clean_dict)
                valid_dict["action_sequence"] = [act_dict]

                f.write(cmd + "\t" + json.dumps(valid_dict) + "\n")
                continue

            if clean_dict["action_type"] == "noop":
                f.write(cmd + "\t" + json.dumps(clean_dict) + "\n")
                print("NOOP")
                continue

            if clean_dict["action_type"] == "composite_action":
                print("COMPOSITE_ACTION")
                f.write(cmd + "\t" + json.dumps(a_dict) + "\n")
                continue

            if toolB_map and cmd in toolB_map:
                child_dict_all = toolB_map[cmd]
                # update action dict with all children except for reference object
                for k, v in child_dict_all.items():
                    if k not in clean_dict:
                        print("BUGGGG")
                    if type(v) == str:
                        v = json.loads(v)
                    if not v:
                        continue
                    if (k in v) and ("reference_object" in v[k]):
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

            if "dialogue_target" in clean_dict and clean_dict["dialogue_target"] == "f1":
                filters_sub_dict = clean_dict["filters"]
                if "team" in filters_sub_dict:
                    new_filters = filters_sub_dict["team"]
                    del filters_sub_dict["team"]
                    filters_sub_dict = new_filters
                if "where_clause" in filters_sub_dict:
                    if len(filters_sub_dict["where_clause"]) > 1 and (not contiguous_spans(filters_sub_dict["where_clause"])):
                        # OR
                        triples = []
                        for item in filters_sub_dict["where_clause"]:
                            triples.append({"pred_text": "has_name", "obj_text": [item]})
                        clean_dict["filters"]["where_clause"] = {"OR": triples}
                    else: # length 1, >1 but contiguous
                        # AND
                        clean_dict["filters"]["where_clause"] = {
                            "AND": [
                                {"pred_text": "has_name",
                                 "obj_text": [merge_indices(filters_sub_dict["where_clause"])]
                                 }]
                        }
            actual_dict = copy.deepcopy(clean_dict)

            action_type = actual_dict["action_type"]

            valid_dict = {}
            valid_dict["dialogue_type"] = actual_dict["dialogue_type"]
            del actual_dict["dialogue_type"]
            actual_dict["action_type"] = actual_dict["action_type"].upper()
            act_dict = fix_spans(actual_dict)
            if 'repeat' in act_dict:
                if act_dict["repeat"]["repeat_key"] == "FOR":
                    valid_dict["remove_condition"] = {'condition': {'comparison_type': 'EQUAL',
                                                   'input_left': {'filters': {'output': {'attribute': 'RUN_COUNT'},
                                                                              'special': {'fixed_value': 'THIS'}}},
                                                   'input_right': {'value': act_dict["repeat"]["repeat_count"]}},
                                                    'condition_type': 'COMPARATOR'}
                elif act_dict["repeat"]["repeat_key"] == "ALL":
                    if "reference_object" not in act_dict:
                        act_dict["reference_object"] = {}
                    if "filters" not in act_dict["reference_object"]:
                        act_dict["reference_object"]["filters"] = {}
                    if "selector" not in act_dict["reference_object"]["filters"]:
                        act_dict["reference_object"]["filters"]["selector"] = {}
                    act_dict["reference_object"]["filters"]["selector"]["return_quantity"] = "ALL"
                act_dict.pop("repeat", None)
            valid_dict["action_sequence"] = [act_dict]

            f.write(cmd + "\t" + json.dumps(valid_dict) + "\n")


def postprocess_step(combined_path, post_processed_path):
    with open(combined_path) as f, open(post_processed_path, 'w') as f_w:
        for line in f.readlines():
            line = line.strip()
            text, d = line.split("\t")
            d = json.loads(d)
            action_dict = d['action_sequence'][0]
            action_type = action_dict['action_type']
            if action_type == 'TAG':
                updated_dict = fix_put_mem(action_dict)
                new_d = {'dialogue_type': d['dialogue_type']}
                new_d.update(updated_dict)
            elif action_type == 'ANSWER':
                new_d = {'dialogue_type': 'GET_MEMORY'}
            elif action_type == "NOOP":
                new_d = {'dialogue_type': action_type}
            else:
                if action_type == 'COPY':
                    action_dict['action_type'] = 'BUILD'
                d['action_sequence'] = [action_dict]
                if "dialogue_target" in action_dict:
                    if action_dict["dialogue_target"] == "f1":
                        if "selector" in action_dict["filters"]:
                            d["dialogue_target"] = {"filters": action_dict["filters"]}
                            del action_dict["filters"]
                            d["dialogue_target"]["filters"]["selector"]["location"] = {"location_type": "SPEAKER_LOOK"}
                            d["dialogue_target"]["filters"]["selector"]["same"] = "DISALLOWED"
                        else:
                            # where clause
                            d["dialogue_target"] = {"filters" : action_dict["filters"]}
                            del action_dict["filters"]
                    # elif new_d["dialogue_target"] == "SWARM":
                    #     d["dialogue_target"]
                    #     new_d["filters"] = {
                    #         "selector": {
                    #             "return_quantity": "ALL",
                    #             "same": "DISALLOWED"
                    #         },
                    #         "memory_type": "AGENT"}
                    # elif new_d["dialogue_target"] == "AGENT":
                    #     new_d["filters"] = {
                    #         "memory_type": "AGENT",
                    #         "selector": {"location": "SPEAKER_LOOK"}
                    #     }
                    del action_dict["dialogue_target"]
                new_d = d
            print(text)
            pprint(new_d)
            print("*" * 40)
            f_w.write(text + "\t" + json.dumps(new_d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default to directory of script being run for writing inputs and outputs
    default_write_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--write_dir_path", type=str, default=default_write_dir)
    args = parser.parse_args()

    # This must exist since we are using tool A outputs
    folder_name_A = '{}/A/all_agreements.txt'.format(args.write_dir_path)
    folder_name_B = '{}/B/all_agreements.txt'.format(args.write_dir_path)
    folder_name_C = '{}/C/all_agreements.txt'.format(args.write_dir_path)
    folder_name_D = '{}/D/all_agreements.txt'.format(args.write_dir_path)
    all_combined_path = '{}/all_combined.txt'.format(args.write_dir_path)
    postprocessed_path = '{}/final_dict_postprocessed.txt'.format(args.write_dir_path)

    collect_tool_outputs(folder_name_C, folder_name_D)
    combine_tool_cd_make_ab(folder_name_A, folder_name_B)
    update_action_dictionaries(all_combined_path)
    postprocess_step(all_combined_path, postprocessed_path)
