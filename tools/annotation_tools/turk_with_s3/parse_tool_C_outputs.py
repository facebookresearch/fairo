import csv
import argparse
import json
from collections import defaultdict, Counter
import re
from operator import itemgetter
from pprint import pprint
import ast

MAX_WORDS = 40


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]


# convert s to snake case
def snake_case(s):
    return re.sub("([a-z])([A-Z])", "\\1_\\2", s).lower()


"""this function splits the key that starts with a given prefix and only for values that are not None
and makes the key be the thing after prefix
"""


def with_prefix(d, prefix):
    new_d = {}
    for k, v in d.items():
        if k.startswith(prefix) and v not in ("", None, "None"):
            index = k.find(prefix) + len(prefix)
            new_key = k[index:]
            new_d[new_key] = v
    return new_d


def fix_nums_at_end_of_ref_obj(d, ps):
    for p in ps:
        d = d.copy()
        rm_keys = []
        add_items = []
        for k, v in d.items():
            new_k = k.split(".")
            updated_ks = []
            for k1 in new_k:
                if k1.startswith(p) and k1[-1].isdigit():
                    rm_keys.append(k)
                    new_k1 = k1[:-1]
                    updated_ks.append(new_k1)
                else:
                    updated_ks.append(k1)
            add_items.append((".".join(updated_ks), v))
        for k in rm_keys:
            del d[k]
        for k, v in add_items:
            d[k] = v
    return d


""" this function removes certain prefixes from keys and renames the key to be: key with text following 
the prefix in the dict"""


def remove_key_prefixes(d, ps):

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
            "name_check.",
            "size_check.",
            "colour_check.",
            "block_type_check.",
            "arg_check.",
            "height_check.",
            "length_check.",
            "width_check.",
            "tag_check.",
            "thickness_check.",
            "coref_resolve_check.yes.",
            "coref_resolve_check.no.",
            "depth_check.",
            "reward.",
        ],
    )
    if "location" in d:
        # fix location type
        r["location"] = {"location_type": d["location"]}

        # fix relative direction
        reference_location_keys = [
            "location.REFERENCE_OBJECT.relative_direction",
            "location.SPEAKER_LOOK_REL.relative_direction",
            "location.SPEAKER_POS_REL.relative_direction",
            "location.AGENT_POS_REL.relative_direction",
        ]
        if any(x in reference_location_keys for x in d.keys()):
            for k, v in d.items():
                if k in reference_location_keys:
                    r["location"]["relative_direction"] = d.get(k)
                    d[k] = None

        # fix steps
        if (k.startswith("location.REFERENCE_OBJECT.steps") for k, v in d.items()):
            new_d = {}
            for k, v in d.items():
                if k.startswith("location.REFERENCE_OBJECT.steps"):
                    parts = k.split(".")
                    new_l = [parts[0]]
                    new_l.extend(parts[2:])
                    new_key = ".".join(new_l)
                    new_d[new_key] = v
                else:
                    new_d[k] = v

            d = new_d

        if r["location"]["location_type"] in [
            "AGENT_POS_REL",
            "SPEAKER_POS_REL",
            "SPEAKER_LOOK_REL",
        ]:
            r["location"]["location_type"] = "".join(r["location"]["location_type"][0:-4])

        if r["location"]["location_type"] == "CONTAINS_COREFERENCE":
            del r["location"]["location_type"]
            r["location"]["contains_coreference"] = "yes"
            r["location"].update(process_dict(with_prefix(d, "location.")))
        elif r["location"]["location_type"] == "coordinates_check":
            r["location"]["location_type"] = "COORDINATES"
            r["location"].update(process_dict(with_prefix(d, "location.")))
        elif r["location"]["location_type"] == "coref_resolve_check":
            del r["location"]["location_type"]
            r["location"].update(process_dict(with_prefix(d, "location.")))
        elif r["location"]["location_type"] == "REFERENCE_OBJECT":
            # here just get span of reference object
            r["location"]["location_type"] = "REFERENCE_OBJECT"
            # update steps in old data

            if "relative_direction" in r["location"]:
                x = process_dict(
                    with_prefix(
                        d,
                        "location.REFERENCE_OBJECT.relative_direction.{}.".format(
                            r["location"]["relative_direction"]
                        ),
                    )
                )
                r["location"].update(x)
                dirn = r["location"]["relative_direction"]
                for k, v in d.items():
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object.has_name.".format(
                            dirn
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object.location.".format(
                            dirn
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object.contains_coreference".format(
                            dirn
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object_1.has_name.".format(
                            r["location"]["relative_direction"]
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object_1.contains_coreference".format(
                            r["location"]["relative_direction"]
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object_2.has_name.".format(
                            r["location"]["relative_direction"]
                        )
                    ):
                        d[k] = None
                    if k.startswith(
                        "location.REFERENCE_OBJECT.relative_direction.{}.reference_object_2.contains_coreference".format(
                            r["location"]["relative_direction"]
                        )
                    ):
                        d[k] = None
            else:
                del r["location"]["location_type"]
            # no key for EXACT
        if ("relative_direction" in r["location"]) and (
            r["location"]["relative_direction"] in ("EXACT", "Other")
        ):
            del r["location"]["relative_direction"]

    for k, v in d.items():

        if (
            k == "location"
            or k in ["COPY"]
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
        elif k == "reference_object" and v == "contains_coreference.yes":
            r["reference_object"] = {"contains_coreference": "yes"}

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


def handle_components(d, child_name):
    output = {}
    # only handle reference_object
    if child_name == "filters":
        output["filters"] = {"reference_object": {}}
        if any(
            k.startswith("reference_object") and v == "contains_coreference.yes"
            for k, v in d.items()
        ):
            output["filters"]["reference_object"]["contains_coreference"] = "yes"

        child_d = process_dict(with_prefix(d, "{}.".format("reference_object")))
        output["filters"]["reference_object"].update(child_d)
    # ref object with coreference
    elif child_name == "reference_object" and any(
        k.startswith("reference_object") and v == "contains_coreference.yes" for k, v in d.items()
    ):
        output[child_name] = {"contains_coreference": "yes"}
        child_d = process_dict(with_prefix(d, "{}.".format(child_name)))
        output[child_name].update(child_d)
    elif child_name == "reference_object" and any(
        x in ["special_reference.SPEAKER", "special_reference.AGENT"] for x in d.values()
    ):
        if "special_reference.SPEAKER" in d.values():
            output[child_name] = {"special_reference": "SPEAKER"}
        elif "special_reference.AGENT" in d.values():
            output[child_name] = {"special_reference": "AGENT"}
        child_d = process_dict(with_prefix(d, "{}.".format(child_name)))
        output[child_name].update(child_d)
    else:
        child_d = process_dict(with_prefix(d, "{}.".format(child_name)))

        if "location" in child_d and "location_type" in child_d["location"]:
            value = child_d["location"]["location_type"]
            child_d["location"].pop("location_type")
            if value in ["SPEAKER_LOOK", "AGENT_POS", "SPEAKER_POS", "COORDINATES"]:
                updated_value = value  # same for coordinates and speaker_look
                if value == "AGENT_POS":
                    updated_value = "AGENT"
                elif value == "SPEAKER_POS":
                    updated_value = "SPEAKER"
                elif value == "COORDINATES":
                    if "coordinates" in child_d["location"]:
                        updated_value = {"coordinates_span": child_d["location"]["coordinates"]}
                    else:
                        updated_value = None

                # add to reference object instead
                if updated_value == None:
                    del child_d["location"]
                else:
                    if "reference_object" in child_d["location"]:
                        child_d["location"]["reference_object"][
                            "special_reference"
                        ] = updated_value
                    else:
                        child_d["location"]["reference_object"] = {
                            "special_reference": updated_value
                        }

                    if "coordinates" in child_d["location"]:
                        del child_d["location"]["coordinates"]

        output[child_name] = {}
        for k, v in child_d.items():
            flag = "yes"
            if k == "comparison":
                flag = "no"
            output[child_name][k] = [flag, child_d[k]]

    return output


def process_result(full_d):
    worker_id = full_d["WorkerId"]
    action_name = full_d["Input.intent"]
    child_name = full_d["Input.child"]
    d = with_prefix(
        full_d, "Answer.root." + action_name + "!" + child_name + "."
    )  # replace with "Answer.root."
    d = fix_nums_at_end_of_ref_obj(d, ["reference_object"])
    receiver_flag = False
    original_child_name = child_name
    if child_name in ["receiver_reference_object", "source_reference_object"]:
        child_name = "reference_object"
        receiver_flag = True
    action_dict = handle_components(d, child_name)
    if receiver_flag:
        action_dict[original_child_name] = action_dict[child_name]
        action_dict.pop(child_name)
    # Fix empty words messing up spans
    #     words = [full_d["Input.word{}".format(x)] for x in range(MAX_WORDS)]
    #     action_dict, words = fix_spans_due_to_empty_words(action_dict, words)
    words = []
    for key in full_d:
        if "Input.word" in key:
            words.append(full_d[key])
    return worker_id, action_dict, words, original_child_name


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
                    # span[0] and span[1] are the same
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
                    # span[0] and span[1] are the same
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


def resolve_spans(words, dicts):
    result = {}
    for d, val in dicts.items():
        new_d = {}
        d = ast.literal_eval(d)
        for k1, v1 in d.items():
            inner = {}
            for k, v in v1.items():
                if type(v) == list:
                    new_v = []
                    for item in v:
                        if item[0] == item[1]:
                            new_v.append(words[item[0]])
                    inner[k] = new_v
                elif k == "repeat":

                    if "stop_condition" in v:
                        new_v = {}
                        new_v["stop_condition"] = {}
                        x = {}
                        x["condition_type"] = v["stop_condition"]["condition_type"]

                        new_vals = []
                        if (
                            v["stop_condition"]["block_type"][0]
                            == v["stop_condition"]["block_type"][1]
                        ):
                            new_vals.append(words[v["stop_condition"]["block_type"][0]])
                        else:
                            for item in v["stop_condition"]["block_type"]:
                                new_vals.append(words[item])
                        x["block_type"] = new_vals
                        new_v["stop_condition"] = x
                        inner["repeat"] = new_v
                else:
                    inner[k] = v
            new_d[k1] = inner
        result[str(new_d)] = val
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_name",
        default="/Users/rebeccaqian/minecraft/tools/annotation_tools/turk_with_s3/C/",
    )
    opts = parser.parse_args()
    folder_name = opts.folder_name
    x = {"reference_object.name_check.has_name.span#3": "on", "reference_object2": "name_check"}
    fix_nums_at_end_of_ref_obj(x, ["reference_object"])

    # convert csv to txt
    result_dict = {}
    f_name = folder_name + "../processed_outputs.csv"

    only_show_disagreements = True
    sentence_mapping = {}
    with open(f_name, "r") as f:
        r = csv.DictReader(f)
        for i, d in enumerate(r):

            sentence = d["Input.command"]
            """ the sentence has a span in it"""
            worker_id, action_dict, words, child_name = process_result(d)

            print(sentence)
            pprint(action_dict)
            print("*" * 20)

            if action_dict is None:
                continue
            command = " ".join(words)
            command = command + "$$$" + child_name

            # if command == "build a statue four feet to your left$$$location":
            sentence_mapping[command] = sentence
            result = json.dumps(action_dict)

            if command in result_dict:
                if len(result_dict[command]) == 3:
                    continue
                result_dict[command].append(result)
            else:
                result_dict[command] = [result]
        print(len(result_dict.keys()))

    # write to txt
    f_name = folder_name + "out.txt"
    with open(f_name, "w") as outfile:
        for k, v in result_dict.items():
            cmd, child = k.split("$$$")
            # len(v) is number of annotations for a command

            if len(v) != 3:
                items = v[0] + "\t" + v[0] + "\t" + v[0]
            else:
                items = "\t".join(v)
            outfile.write(cmd + "\t" + child + "\t" + items + "\n")

    # construct counter from txt
    result_counts = defaultdict(Counter)
    f_name = folder_name + "out.txt"
    with open(f_name) as in_data:
        for line in in_data.readlines():
            line = line.strip()
            cmd, child, r1, r2, r3 = line.split("\t")
            for r in [r1, r2, r3]:
                r_new = remove_definite_articles(cmd, r)
                result_counts[cmd + "$$$" + child][json.dumps(r_new)] += 1
    print(len(result_counts.keys()))

    # compute agreements and disagreements
    no_agreement = 0
    num_agreements = 2
    agreement = 0
    only_show_disagreements = False
    disagreement = defaultdict(Counter)
    all_agreements_dict = {}
    for command, counts in sorted(result_counts.items()):
        if not any(v >= num_agreements for v in counts.values()):
            if only_show_disagreements:
                print(command)
            disagreement[command] = counts
            no_agreement += 1
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
    print(folder_name)
    ag = str(agreement)
    f = folder_name + ag + "_agreements.txt"
    with open(f, "w") as outfile:
        for k, v in all_agreements_dict.items():
            cmd, child = k.split("$$$")
            outfile.write(cmd + "\t" + child + "\t" + v + "\n")

    # write disagreements to a file
    disag = str(no_agreement)
    f = folder_name + disag + "_disagreements.txt"
    with open(f, "w") as outfile:
        for k, v in disagreement.items():
            cmd, child = k.split("$$$")
            outfile.write(cmd + "\t" + child + "\n")
            for item in v:
                outfile.write(item + "\n")
            outfile.write("\n")
            outfile.write("\n")

    with open(folder_name + "/all_agreements.txt", "w") as f, open(
        folder_name + ag + "_agreements.txt"
    ) as f1, open(folder_name + disag + "_disagreements.txt") as f2:
        for line in f1.readlines():
            line = line.strip()
            f.write(line + "\n")
        for line in f2.readlines():
            line = line.strip()
            f.write(line + "\n")

    for command, counts in disagreement.items():
        words = command.split()
        parts = words[-1].split("$$$")
        print(sentence_mapping[command])
        words[-1] = parts[0]
        child_name = parts[1]
        command = " ".join(words)
        c = resolve_spans(words, counts)
        print(command, child_name)
        for k, v in c.items():
            pprint(ast.literal_eval(k))
            print("-" * 10)
        print("*" * 30)
