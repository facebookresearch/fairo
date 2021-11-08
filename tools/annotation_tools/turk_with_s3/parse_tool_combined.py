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
        return {"remove_condition": {"condition_type": "NEVER"}}
    if d["loop"] == "repeat_until":
        stripped_d = with_prefix(d, "loop.repeat_until.")
        if not stripped_d:
            return None
        processed_d = process_dict(stripped_d)
        if "adjacent_to_block_type" in processed_d:
            return {
                "remove_condition": {
                    "condition_type": "ADJACENT_TO_BLOCK_TYPE",
                    "block_type": processed_d["adjacent_to_block_type"],
                }
            }
        elif "condition_span" in processed_d:
            return {"remove_condition": {"condition_span": processed_d["condition_span"]}}

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
            "dialogue_target.f1.",
            "dialogue_target.f2.",
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
        if k in ["target_action_type", "has_block_type", "dance_type_name", "tag_val", "dance_type_span"]:
            output[k] = ["yes", v]

        elif type(v) == list or (k == "receiver"):
            output[k] = ["no", v]
        else:
            output[k] = ["yes", v]
    return output


def process_result(full_d):
    worker_id = full_d["WorkerId"]
    answers = []
    for i in range(1, 6):
        if full_d["Answer.command_"+ str(i)]:
            answers.append(full_d["Answer.command_"+ str(i)])

    return worker_id, answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="processed_outputs_combined.csv"
    )
    opts = parser.parse_args()
    tokenizer = English().Defaults.create_tokenizer()

    # convert csv to txt first
    num_agreements = 2
    result_dict = {}
    f_name = opts.input_file
    only_show_disagreements = True
    with open(f_name, "r") as f:
        r = csv.DictReader(f)
        for i, d in enumerate(r):
            worker_id = d["WorkerId"]
            sentence = d["Input.sentence"]
            worker_id, answers = process_result(d)
            print("Original sentence: %r" % sentence)
            print("Split commands: %r " % (" , ".join(answers)))
            print("*" * 20)
            if sentence in result_dict:
                print("DUPLICATES!!!")
                print(sentence)
            else:
                result_dict[sentence] = answers

    print(len(result_dict.keys()))

    # write to txt
    f_name = "combined_out.txt"
    # write out combined mapping, add to input.txt
    with open(f_name, "w") as outfile, open("input.txt", "a") as inp_file:
        for k, v in result_dict.items():
            cmd = k
            all_ans = ""
            for answer in v:
                inp_file.write(answer + "\n")
                all_ans = all_ans + answer + "\t"
            outfile.write(cmd + "\t" + all_ans.strip() + "\n")