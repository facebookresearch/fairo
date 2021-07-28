"""
Copyright (c) Facebook, Inc. and its affiliates.

This file has functions to preprocess the chat from user before
querying the dialogue manager
"""
import ast
import csv
import argparse
import json
from collections import defaultdict, Counter
import re
import string
from spacy.lang.en import English
from typing import List
from operator import itemgetter
from pprint import pprint
import os


def word_tokenize(st) -> str:
    chat_with_spaces = insert_spaces(st)
    return " ".join([str(x) for x in tokenizer(chat_with_spaces)])


def sentence_split(st):
    st = st.replace(" ?", " .")
    st = st.replace(" !", " .")
    st = st.replace(" ...", " .")
    res = [
        " ".join([x for x in sen.lower().split() if x not in string.punctuation])
        for sen in st.split(" .")
    ]
    return [x for x in res if x != ""]


def insert_spaces(chat):
    updated_chat = ""
    for i, c in enumerate(chat):
        # [num , (num , {num , ,num , :num
        if (
            (c in ["[", "(", "{", ",", ":", "x"])
            and (i != len(chat) - 1)
            and (chat[i + 1].isdigit())
        ):
            updated_chat += c + " "
        # num, , num] , num) , num}, num:
        # 4x -> 4 x
        elif (
            (c.isdigit())
            and (i != len(chat) - 1)
            and (chat[i + 1] in [",", "]", ")", "}", ":", "x"])
        ):
            updated_chat += c + " "
        else:
            updated_chat += c

    return updated_chat


def preprocess_chat(chat: str) -> List[str]:
    # For debug mode, return as is.
    if chat == "_debug_" or chat.startswith("_ttad_"):
        return [chat]

    # Tokenize
    tokenized_line = word_tokenize(chat)
    tokenized_sentences = [sen for sen in sentence_split(tokenized_line)]

    return tokenized_sentences


def process_repeat_dict(d):
    if d["loop"] == "ntimes":
        repeat_dict = {"repeat_key": "FOR"}
        processed_d = process_dict(with_prefix(d, "loop.ntimes."))
        if "repeat_for" in processed_d:
            repeat_dict["repeat_count"] = processed_d["repeat_for"]
        if "repeat_dir" in processed_d:
            repeat_dict["repeat_dir"] = processed_d["repeat_dir"]
        return repeat_dict
    if d["loop"] == "repeat_all":
        repeat_dict = {"repeat_key": "ALL"}
        processed_d = process_dict(with_prefix(d, "loop.repeat_all."))
        if "repeat_dir" in processed_d:
            repeat_dict["repeat_dir"] = processed_d["repeat_dir"]
        return repeat_dict
    if d["loop"] == "forever":
        return {"stop_condition": {"condition_type": "NEVER"}}
    if d["loop"] == "repeat_until":
        stripped_d = with_prefix(d, "loop.repeat_until.")
        if not stripped_d:
            return None
        processed_d = process_dict(stripped_d)
        if "adjacent_to_block_type" in processed_d:
            return {
                "stop_condition": {
                    "condition_type": "ADJACENT_TO_BLOCK_TYPE",
                    "block_type": processed_d["adjacent_to_block_type"],
                }
            }
        elif "condition_span" in processed_d:
            return {"stop_condition": {"condition_span": processed_d["condition_span"]}}

    raise NotImplementedError("Bad repeat dict option: {}".format(d["loop"]))


def process_get_memory_dict(d):
    filters_val = d["filters"]
    out_dict = {"filters": {}}
    parent_dict = {}
    if filters_val.startswith("type."):
        parts = remove_prefix(filters_val, "type.").split(".")
        type_val = parts[0]
        if type_val in ["ACTION", "AGENT"]:
            out_dict["filters"]["temporal"] = "CURRENT"
            tag_val = parts[1]
            out_dict["answer_type"] = "TAG"
            out_dict["tag_name"] = parts[1]  # the name of tag is here
            if type_val == "ACTION":
                x = with_prefix(d, "filters." + filters_val + ".")
                out_dict["filters"].update(x)
        elif type_val in ["REFERENCE_OBJECT"]:
            d.pop("filters")
            ref_obj_dict = remove_key_prefixes(d, ["filters.type."])
            ref_dict = process_dict(ref_obj_dict)
            if "answer_type" in ref_dict["reference_object"]:
                out_dict["answer_type"] = ref_dict["reference_object"]["answer_type"]
                ref_dict["reference_object"].pop("answer_type")
            if "tag_name" in ref_dict["reference_object"]:
                out_dict["tag_name"] = ref_dict["reference_object"]["tag_name"]
                ref_dict["reference_object"].pop("tag_name")
            out_dict["filters"].update(ref_dict)

        out_dict["filters"]["type"] = type_val

    return out_dict


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]


def handle_get_memory(d):
    out_d = {"dialogue_type": "GET_MEMORY"}
    child_d = process_get_memory_dict(with_prefix(d, "action_type.ANSWER."))
    out_d.update(child_d)
    return out_d


# convert s to snake case
def snake_case(s):
    return re.sub("([a-z])([A-Z])", "\\1_\\2", s).lower()


def with_prefix(d, prefix):
    """
    This function splits the key that starts with a given prefix and only for values that are not None
    and makes the key be the thing after prefix
    """
    return {
        k.split(prefix)[1]: v
        for k, v in d.items()
        if k.startswith(prefix) and v not in ("", None, "None")
    }


def remove_key_prefixes(d, ps):
    """This function removes certain prefixes from keys and renames the key to be: key with text following
    the prefix in the dict.
    """
    for p in ps:
        d = d.copy()
        rm_keys = []
        add_items = []
        for k, v in d.items():
            if k.startswith(p):
                rm_keys.append(k)
                add_items.append((k[len(p) :], v))
        for k in rm_keys:
            del d[k]
        for k, v in add_items:
            d[k] = v
    return d


def fix_spans_due_to_empty_words(action_dict, words):
    """Return modified (action_dict, words)"""

    def reduce_span_vals_gte(d, i):
        for k, v in d.items():
            if type(v) == dict:
                reduce_span_vals_gte(v, i)
                continue
            try:
                a, b = v
                if a >= i:
                    a -= 1
                if b >= i:
                    b -= 1
                d[k] = [[a, b]]
            except ValueError:
                pass
            except TypeError:
                pass

    # remove trailing empty strings
    while words[-1] == "":
        del words[-1]

    # fix span
    i = 0
    while i < len(words):
        if words[i] == "":
            reduce_span_vals_gte(action_dict, i)
            del words[i]
        else:
            i += 1

    return action_dict, words


def process_dict(d):
    r = {}
    d = remove_key_prefixes(
        d,
        [
            "TURN_CHECK.LOOK.",
            "TURN_CHECK.POINT.",
            "TURN_CHECK.TURN.",
            "MOVE.yes.",
            "MOVE.no.",
            "COPY.yes.",
            "COPY.no.",
            "receiver_loc.",
            "receiver_ref.",
            "source_loc.",
            "source_ref.",
            "FREEBUILD.BUILD.",
            "answer_type.TAG.",
            "FREEBUILD.FREEBUILD.",
            "coref_resolve_check.yes.",
            "coref_resolve_check.no.",
        ],
    )
    if "location" in d:
        r["location"] = {"location_type": d["location"]}
        if r["location"]["location_type"] == "coref_resolve_check":
            del r["location"]["location_type"]
        elif r["location"]["location_type"] == "REFERENCE_OBJECT":
            r["location"]["location_type"] = "REFERENCE_OBJECT"
            r["location"]["relative_direction"] = d.get(
                "location.REFERENCE_OBJECT.relative_direction"
            )
            # no key for EXACT
            if r["location"]["relative_direction"] in ("EXACT", "Other"):
                del r["location"]["relative_direction"]
            d["location.REFERENCE_OBJECT.relative_direction"] = None
        r["location"].update(process_dict(with_prefix(d, "location.")))

    for k, v in d.items():
        if (
            k == "location"
            or k in ["COPY", "coref_resolve_check", "receiver", "source"]
            or (k == "relative_direction" and v in ("EXACT", "NEAR", "Other"))
        ):
            continue
        # handle span
        if re.match("[^.]+.span#[0-9]+", k):
            prefix, rest = k.split(".", 1)
            idx = int(rest.split("#")[-1])
            if prefix in r:
                r[prefix].append([idx, idx])
                r[prefix] = sorted(r[prefix], key=itemgetter(0))
            else:
                r[prefix] = [[idx, idx]]

        # handle nested dict
        elif "." in k:
            prefix, rest = k.split(".", 1)
            prefix_snake = snake_case(prefix)
            r[prefix_snake] = r.get(prefix_snake, {})
            r[prefix_snake].update(process_dict(with_prefix(d, prefix + ".")))

        # handle const value
        else:
            r[k] = v
    return r


def handle_put_memory(d):
    return {}


def handle_commands(d):
    output = {}
    action_name = d["action_type"]
    formatted_dict = with_prefix(d, "action_type.{}.".format(action_name))
    child_d = process_dict(with_prefix(d, "action_type.{}.".format(action_name)))
    # Fix Build/Freebuild mismatch
    if child_d.get("FREEBUILD") == "FREEBUILD":
        action_name = "FREEBUILD"
    child_d.pop("FREEBUILD", None)

    if "MOVE" in child_d:
        if child_d["MOVE"] == "yes":
            action_name = "MOVE"
        elif child_d["MOVE"] == "no":
            action_name = "DANCE"
        child_d.pop("MOVE")

    if formatted_dict.get("COPY", "no") == "yes":
        action_name = "COPY"
        formatted_dict.pop("COPY")

    # add action type info
    if "TURN_CHECK" in child_d:
        output["action_type"] = ["yes", child_d["TURN_CHECK"].lower()]
        child_d.pop("TURN_CHECK")
    else:
        output["action_type"] = ["yes", action_name.lower()]
    # add dialogue type info
    if output["action_type"][1] == "tag":
        output["dialogue_type"] = ["yes", "PUT_MEMORY"]
    else:
        output["dialogue_type"] = ["yes", "HUMAN_GIVE_COMMAND"]

    if output["action_type"][1] == "get":
        if "receiver" in child_d:
            if "reference_object" in child_d["receiver"]:
                child_d["receiver_reference_object"] = child_d["receiver"]["reference_object"]
            elif "location" in child_d["receiver"]:
                child_d["receiver_location"] = child_d["receiver"]["location"]
            child_d.pop("receiver")
        if "source" in child_d:
            if "reference_object" in child_d["source"]:
                child_d["source_reference_object"] = child_d["source"]["reference_object"]
            elif "location" in child_d["source"]:
                child_d["source_location"] = child_d["source"]["location"]
            child_d.pop("source")

    for k, v in child_d.items():
        if k in ["target_action_type", "has_block_type", "dance_type_name"]:
            output[k] = ["yes", v]

        elif type(v) == list or (k == "receiver"):
            output[k] = ["no", v]
        else:
            output[k] = ["yes", v]
    return output


def process_result(full_d):

    worker_id = full_d["WorkerId"]
    d = with_prefix(full_d, "Answer.root.")
    if not d:
        return worker_id, {}, full_d["Input.command"].split()
    try:
        action = d["action_type"]
    except KeyError:
        return worker_id, {}, full_d["Input.command"].split()

    action_dict = handle_commands(d)

    ##############
    # repeat dict
    ##############
    # NOTE: this can probably loop over or hold indices of which specific action ?
    if action_dict.get("dialogue_type", [None, None])[1] == "HUMAN_GIVE_COMMAND":
        if d.get("loop") not in [None, "Other"]:
            repeat_dict = process_repeat_dict(d)
            if repeat_dict:
                # Some turkers annotate a repeat dict for a repeat_count of 1.
                # Don't include the repeat dict if that's the case
                if repeat_dict.get("repeat_dir", None) == "Other":
                    repeat_dict.pop("repeat_dir")
                if repeat_dict.get("repeat_count"):
                    a, b = repeat_dict["repeat_count"][0]
                    repeat_count_str = " ".join(
                        [full_d["Input.word{}".format(x)] for x in range(a, b + 1)]
                    )
                    if repeat_count_str not in ("a", "an", "one", "1"):
                        action_dict["repeat"] = ["yes", repeat_dict]
                else:
                    action_dict["repeat"] = ["yes", repeat_dict]

    ##################
    # post-processing
    ##################
    # Fix empty words messing up spans
    words = []
    for key in full_d:
        if "Input.word" in key:
            words.append(full_d[key])

    return worker_id, action_dict, words


def fix_cnt_in_schematic(words, action_dict):
    if "repeat" not in action_dict:
        return action_dict
    repeat = action_dict["repeat"]
    val = []
    if "repeat_count" in repeat[1]:
        val = repeat[1]["repeat_count"]
    elif "repeat_key" in repeat[1] and repeat[1]["repeat_key"] == "ALL":
        if any(x in ["all", "every", "each"] for x in words):
            if "all" in words:
                all_val = words.index("all")
            elif "each" in words:
                all_val = words.index("each")
            elif "every" in words:
                all_val = words.index("every")
            val = [[all_val, all_val]]
    else:
        return action_dict

    for k, v in action_dict.items():
        if k in ["schematic", "reference_object"]:
            for i, meh in enumerate(v[1]):
                if meh in val:
                    v[1].pop(i)
            action_dict[k] = [v[0], v[1]]
    return action_dict


def remove_definite_articles(cmd, d):
    words = cmd.split()
    if type(d) == str:
        d = ast.literal_eval(d)
    new_d = {}
    for k, v in d.items():
        # for level 1
        if type(v) == list and v[0] in ["yes", "no"]:
            if type(v[1]) == list:
                new_v = []
                for span in v[1]:
                    if words[span[0]] in ["the", "a", "an"]:
                        continue
                    new_v.append(span)
                new_d[k] = [v[0], new_v]
            elif type(v[1]) == dict:
                v_new = remove_definite_articles(cmd, v[1])
                new_d[k] = [v[0], v_new]

            else:
                new_d[k] = v
        # for recursion on normal internal dict
        else:
            if type(v) == list:
                new_v = []
                for span in v:
                    if words[span[0]] in ["the", "a", "an"]:
                        continue
                    new_v.append(span)
                new_d[k] = new_v
            elif type(v) == dict:
                v_new = remove_definite_articles(cmd, v)
                new_d[k] = v_new

            else:
                new_d[k] = v

    return new_d


# ONLY FOR DEBUGGING
def resolve_spans(words, dicts):
    result = {}
    mapping_old_dicts = {}
    for d, val in dicts.items():
        new_d = {}
        d = ast.literal_eval(d)
        for k, v in d.items():
            if type(v[1]) == list:
                new_v = []
                for item in v[1]:
                    new_v.append(words[item[0]])
                new_d[k] = [v[0], new_v]
            elif k == "repeat":
                # v[1] = ast.literal_eval(v[1])
                if "stop_condition" in v[1]:
                    new_v = {}
                    new_v["stop_condition"] = {}
                    x = {}

                    if "condition_type" in v[1]["stop_condition"]:
                        x["condition_type"] = v[1]["stop_condition"]["condition_type"]
                    new_vals = []
                    if "block_type" in v[1]["stop_condition"]:
                        for item in v[1]["stop_condition"]["block_type"]:
                            new_vals.append(words[item[0]])
                        x["block_type"] = new_vals
                    elif "condition_span" in v[1]["stop_condition"]:
                        for item in v[1]["stop_condition"]["condition_span"]:
                            new_vals.append(words[item[0]])
                        x["condition_span"] = new_vals
                    new_v["stop_condition"] = x
                    new_d["repeat"] = [v[0], new_v]
                else:
                    new_d[k] = v
            else:
                new_d[k] = v
        result[str(new_d)] = val
        mapping_old_dicts[str(new_d)] = str(d)
    return result, mapping_old_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to directory of script being run for writing inputs and outputs
    default_write_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--folder_name",
        default="{}/A/".format(default_write_dir),
    )
    opts = parser.parse_args()
    print(opts)
    tokenizer = English().Defaults.create_tokenizer()

    # convert csv to txt first
    # def process_results_with_agreements(f_name, num_agreements=1, debug=False, tsv=False, only_show_disagreements=False):
    num_agreements = 2
    result_dict = {}
    folder_name = opts.folder_name
    f_name = folder_name + "../A/processed_outputs.csv"
    only_show_disagreements = True
    with open(f_name, "r") as f:
        r = csv.DictReader(f)
        for i, d in enumerate(r):
            worker_id = d["WorkerId"]
            sentence = preprocess_chat(d["Input.command"])[0]
            _, action_dict, words = process_result(d)
            a_dict = fix_cnt_in_schematic(words, action_dict)
            print(sentence)
            pprint(a_dict)
            print("*" * 20)
            if a_dict is None:
                continue
            command = " ".join(words)
            result = json.dumps(a_dict)
            if command in result_dict:
                if len(result_dict[command]) == 3:
                    print(command)
                    continue
                result_dict[command].append(result)
            else:
                result_dict[command] = [result]

    print(len(result_dict.keys()))

    # write to txt
    f_name = folder_name + "out.txt"
    with open(f_name, "w") as outfile:
        for k, v in result_dict.items():
            cmd = k
            if len(v) == 1:
                items = v[0] + "\t" + v[0] + "\t" + v[0]
            else:
                items = "\t".join(v)
            outfile.write(cmd + "\t" + items + "\n")

    # construct counter from txt
    result_counts = defaultdict(Counter)
    f_name = folder_name + "out.txt"
    import ast

    with open(f_name) as in_data:
        for line in in_data.readlines():
            line = line.strip()
            # print(len(line.split("\t")))
            parts = line.split("\t")
            if len(parts) == 4:
                cmd, r1, r2, r3 = parts
            elif len(parts) == 3:  # for just one answer
                cmd, r1, r2 = parts
                r3 = r2
            else:
                cmd, r = parts
                r1, r2, r3 = r, r, r
            for r in [r1, r2, r3]:
                r_new = remove_definite_articles(cmd, r)
                result_counts[cmd][json.dumps(r_new)] += 1
    print(len(result_counts.keys()))

    no_agreement = 0
    agreement = 0
    disagreement = defaultdict(Counter)
    only_show_disagreements = False
    all_agreements_dict = {}
    disagreements_dict = {}

    for command, counts in sorted(result_counts.items()):
        if not any(v >= num_agreements for v in counts.values()):
            if only_show_disagreements:
                print(command)
            disagreement[command] = counts
            no_agreement += 1
            for result, count in counts.items():
                if command not in disagreements_dict:
                    disagreements_dict[command] = [result]
                else:
                    disagreements_dict[command].append(result)

            continue
        elif only_show_disagreements:
            continue

        for result, count in counts.items():
            if count >= num_agreements:
                all_agreements_dict[command] = result
                agreement += 1

        print(agreement)
        print(no_agreement)

    # write out agreements to a file
    ## format is : command child dict
    ag = str(agreement)
    f = folder_name + ag + "_agreements.txt"
    with open(f, "w") as outfile:
        for k, v in all_agreements_dict.items():
            cmd = k
            outfile.write(cmd + "\t" + v + "\n")

    # write disagreements to a file
    disag = str(no_agreement)
    f = folder_name + disag + "_disagreements.txt"
    with open(f, "w") as outfile:
        for k, v in disagreements_dict.items():
            cmd = k
            outfile.write(cmd + "\n")
            for item in v:
                outfile.write(item + "\n")
            outfile.write("\n")

    for command, counts in disagreement.items():
        words = command.split()
        c, mapping_old_dicts = resolve_spans(words, counts)
        print(command)
        for k, v in c.items():
            pprint(ast.literal_eval(k))
        print("*" * 30)

    with open(folder_name + "all_agreements.txt", "w") as f_out, open(
        folder_name + ag + "_agreements.txt"
    ) as f1, open(folder_name + disag + "_disagreements.txt") as f_in:
        for line in f1.readlines():
            cmd, out = line.strip().split("\t")
            cmd = preprocess_chat(cmd)[0]
            f_out.write(cmd + "\t" + out + "\n")
        for line in f_in.readlines():
            cmd, out = line.strip().split("\t")
            f_out.write(cmd + "\t" + out + "\n")
