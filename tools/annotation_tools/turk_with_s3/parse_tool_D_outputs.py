import csv
import argparse
from collections import defaultdict, Counter
import re
from operator import itemgetter
import json
import ast
from pprint import pprint
import os

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


""" check for the following use cases:
1. plain ref object woith coref
2. ref obj with name, color etc
3. ref obj with name , color + filters
4. ref obj with name, filters and location
5. ref obj with filters only
"""


def process_dict(d):
    r = {}
    d = remove_key_prefixes(
        d,
        [
            "RELATIVE_DIRECTION.",
            "distance_to.",
            "NUM_BLOCKS.",
            "colour_check.",
            "ordinal_other.",
            "arg_check_type.ranked.",
            "arg_check_type.fixed.",
            "measure_check.argmin.",
            "measure_check.argmax.",
            "measure_check.greater_than.",
            "measure_check.less_than.",
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

    if ("quantity" in d) and (
        d["quantity"] in ["RELATIVE_DIRECTION", "distance_to", "NUM_BLOCKS"]
    ):
        d["quantity"] = {}  # or d.pop('quantity')

    for k, v in d.items():

        # skip processing these
        if (
            k == "location"
            or k in ["COPY"]
            or k in ["block_filters0"]
            or (k == "relative_direction" and v in ("EXACT", "NEAR", "Other"))
            or (k == "ordinal" and v == "ordinal_other")
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
            print(k, v)
            print(prefix_snake, r[prefix_snake])
            r[prefix_snake].update(process_dict(with_prefix(d, prefix + ".")))

        # handle const value
        else:
            r[k] = v
    return r


def handle_components(d):
    output = {}
    d.pop("arg_check_type", None)
    updated_dict = process_dict(d)
    output = {}
    if "measure_check" in updated_dict:
        ranking_measure = updated_dict["measure_check"]
        updated_dict.pop("measure_check")
        output[ranking_measure] = {"quantity": updated_dict["quantity"]}
        # handle argmax and argmin
        if "ordinal" in updated_dict:
            output[ranking_measure]["ordinal"] = updated_dict["ordinal"]
        # handle greater+_than and less_than
        if "number" in updated_dict:
            output[ranking_measure]["number"] = updated_dict["number"]
        return output

    return output


def process_result(full_d):
    worker_id = full_d["WorkerId"]
    action_child_name = full_d["Input.child"]
    ref_child_name = full_d["Input.ref_child"]
    d = with_prefix(full_d, "Answer.root.")  # replace with "Answer.root."
    print(d)
    action_dict = handle_components(d)

    # Fix empty words messing up spans
    words = []
    for key in full_d:
        if "Input.word" in key:
            words.append(full_d[key])
    return worker_id, action_dict, words


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
    # Default to directory of script being run for writing inputs and outputs
    default_write_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument(
        "--folder_name",
        type=str,
        default="{}/D/".format(default_write_dir),
    )
    opts = parser.parse_args()
    # convert csv to txt
    result_dict = {}
    folder_name = opts.folder_name
    f_name = folder_name + "processed_outputs.csv"
    only_show_disagreements = True
    sentence_mapping = {}
    with open(f_name, "r") as f:
        r = csv.DictReader(f)
        for i, d in enumerate(r):
            sentence = d["Input.command"]
            """ the sentence has a span in it"""

            worker_id, action_dict, words = process_result(d)
            print(sentence)
            pprint(action_dict)
            print("*" * 20)

            if action_dict is None:
                continue

            command = " ".join(words)
            sentence_mapping[command] = sentence
            result = json.dumps(action_dict)
            if command in result_dict:
                if len(result_dict[command]) == 3:
                    continue
                result_dict[command].append(result)
            else:
                result_dict[command] = [result]
    # write to txt
    f_name = folder_name + "out.txt"
    with open(f_name, "w") as outfile:
        for k, v in result_dict.items():
            cmd = k
            # len(v) is number of annotations for a command
            child = "comparison"
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
    f = folder_name + "14_agreements.txt"
    with open(f, "w") as outfile:
        for k, v in all_agreements_dict.items():
            cmd, child = k.split("$$$")
            outfile.write(cmd + "\t" + child + "\t" + v + "\n")

    # write disagreements to a file
    f = folder_name + "0_disagreements.txt"
    with open(f, "w") as outfile:
        for k, v in disagreement.items():
            cmd, child = k.split("$$$")
            outfile.write(cmd + "\t" + child + "\n")
            for item in v:
                outfile.write(item + "\n")
            outfile.write("\n")
            outfile.write("\n")

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

    with open(folder_name + "/all_agreements.txt", "w") as f, open(
        folder_name + "14_agreements.txt"
    ) as f1, open(folder_name + "0_disagreements.txt") as f2:
        for line in f1.readlines():
            line = line.strip()
            f.write(line + "\n")
        for line in f2.readlines():
            line = line.strip()
            f.write(line + "\n")
