"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import copy
import os
import tarfile
import pandas as pd
import json
import boto3
import spacy
import re

spacy_model = spacy.load("en_core_web_sm")

S3_BUCKET_NAME = "droidlet-hitl"
S3_ROOT = "s3://droidlet-hitl"
NSP_OUTPUT_FNAME = "nsp_outputs"
ANNOTATED_COMMANDS_FNAME = "nsp_data.txt"

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
s3 = boto3.client(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def postprocess_logical_form(chat, logical_form):
    """This function performs some postprocessing on the logical form:
    substitutes indices with words and resolves coreference"""
    # perform lemmatization on the chat
    lemmatized_chat = spacy_model(chat)
    lemmatized_chat_str = " ".join(str(word.lemma_) for word in lemmatized_chat)

    # Get the words from indices in spans and substitute fixed_values
    process_spans_and_remove_fixed_value(
        logical_form, re.split(r" +", chat), re.split(r" +", lemmatized_chat_str)
    )

    return logical_form


def process_spans_and_remove_fixed_value(d, original_words, lemmatized_words):
    """This function fetches the words corresponding to indices in the logical form
    and fetches the values of "fixed_value" key in the dictionary in place.
    """
    if type(d) is not dict:
        return
    for k, v in d.items():
        if type(v) == dict:
            # substitute the value of "fixed_value" in place in the dictionary
            if "fixed_value" in v.keys():
                d[k] = v["fixed_value"]
            else:
                process_spans_and_remove_fixed_value(v, original_words, lemmatized_words)
        elif type(v) == list and len(v) > 0 and type(v[0]) == dict:
            # triples
            for a in v:
                process_spans_and_remove_fixed_value(a, original_words, lemmatized_words)
        else:
            try:
                sentence, (L, R) = v
                if sentence != 0:
                    raise NotImplementedError("Must update process_spans for multi-string inputs")
                if L > R:
                    L = R
                if L < 0:
                    L = 0
                if R > (len(lemmatized_words) - 1):
                    R = len(lemmatized_words) - 1
            except ValueError:
                continue
            except TypeError:
                continue
            original_w = " ".join(original_words[L : (R + 1)])
            # The lemmatizer converts 'it' to -PRON-
            if original_w == "it":
                d[k] = original_w
            else:
                d[k] = " ".join(lemmatized_words[L : (R + 1)])


def remove_text_span(d: dict):
    d_copy = copy.deepcopy(d)
    for k, v in d.items():
        if type(v) == dict:
            d_copy[k] = remove_text_span(d_copy[k])
        if type(v) == list:
            for i in range(len(v)):
                if type(v[i]) == dict:
                    d_copy[k][i] = remove_text_span(d_copy[k][i])
        if k == "text_span":
            del d_copy[k]

    return d_copy


def build_comparison_dicts(batch_id):
    # Retrieve all of the log file keys from S3 bucket for the given batch_id
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{batch_id}/interaction")
    log_keys = [item["Key"] for item in response["Contents"]]

    # Download each log file, extract, and pull the errors/commands into a dictionary
    error_dict = {}
    command_dict = {}
    tot_num_errors = 0
    tot_parse_errors = 0
    tot_num_cmds = 0
    for key in log_keys:
        s3.download_file(S3_BUCKET_NAME, key, "logs.tar.gz")
        tf = tarfile.open("logs.tar.gz")
        tf.extract("./error_details.csv")
        csv_file = pd.read_csv("error_details.csv", delimiter="|")
        for idx, row in csv_file.iterrows():
            tot_num_errors += 1
            if row["parser_error"] == True:
                tot_parse_errors += 1
                cmd = row["command"].replace("&nbsp ;", "").replace("?", "").strip().lower()
                if cmd in error_dict:
                    continue
                else:
                    error_dict[cmd] = json.loads(row["action_dict"].replace("'", '"'))

        tf.extract("./nsp_outputs.csv")
        csv_file = pd.read_csv("nsp_outputs.csv", delimiter="|")
        for idx, row in csv_file.iterrows():
            tot_num_cmds += 1
            cmd = row["command"].replace("&nbsp ;", "").replace("?", "").strip().lower()
            if cmd in command_dict:
                continue
            else:
                # error_details.csv has already been postprocessed, but nsp_outputs.csv has not
                command_dict[cmd] = postprocess_logical_form(
                    cmd, json.loads(row["action_dict"].replace("'", '"'))
                )

    # Remove text_span from dicts
    cd_copy = copy.deepcopy(command_dict)
    for key in command_dict.keys():
        cd_copy[key] = remove_text_span(command_dict[key])
    command_dict = copy.deepcopy(cd_copy)

    ed_copy = copy.deepcopy(error_dict)
    for key in error_dict.keys():
        ed_copy[key] = remove_text_span(error_dict[key])
    error_dict = copy.deepcopy(ed_copy)

    print("\n*** Finished building dicts for batch cmd:LF and err:LF ***\n")
    print(f"Total number of commands issued: {tot_num_cmds}")
    print(f"Total number of commands dedup: {len(command_dict)}")
    print(f"Total number of agent errors: {tot_num_errors}")
    print(f"Total number of NSP errors: {tot_parse_errors}")
    print(f"Total number of NSP errors dedup: {len(error_dict)}")
    print("\n")

    return command_dict, error_dict


def compare_against_gt(d: dict, anno_d: dict, label: str):
    nsp_errors = 0
    commands_annotated = 0
    not_found = 0
    for key in d.keys():
        if key not in anno_d:
            not_found += 1
            continue
        if d[key] != anno_d[key]:
            nsp_errors += 1
            # if label == "commands":
            #     print(key)
            #     print(d[key])
            #     print(anno_d[key])
            #     print("\n")
        commands_annotated += 1

    print(f"Num {label} with NO annotated GT: {not_found}")
    print(f"Implied # of {label} annotated (at some point): {commands_annotated}")
    print(f"(Known) NSP Errors in {label}: {nsp_errors}")
    print(f"(Known) NSP Successes in {label}: {commands_annotated - nsp_errors}")
    print("\n")


def build_annotated_dict(nsp_fname: str, scrape_anno_outs: bool):
    # Download nsp_data and build a annotated command dict
    s3.download_file(S3_BUCKET_NAME, nsp_fname, "nsp_data.txt")
    with open("nsp_data.txt", "r") as f:
        nsp_data = f.readlines()
    annotated_dict = {}
    anno_cmd_list = []
    for line in nsp_data:
        split_line = line.strip().split("|")
        cmd_idx = len(split_line) - 2
        cmd = split_line[cmd_idx].replace("&nbsp ;", "").replace("?", "").strip().lower()
        anno_cmd_list.append(cmd)
        annotated_dict[cmd] = postprocess_logical_form(
            cmd, json.loads(split_line[cmd_idx + 1].strip())
        )

    if scrape_anno_outs:
        # Retrieve all of the annotated command file keys from S3 bucket for the given batch_id
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{opts.batch_id}/annotated")
        anno_keys = []
        for page in pages:
            anno_keys += [item["Key"] for item in page["Contents"]]

        print(f"Actual number of commands annotated: {len(anno_keys)}")
        print(
            f"Size of annotated dict before adding batch-specific annotations: {len(annotated_dict)}"
        )

        # Download the annotation txt files and add them to annotated_dict if they don't already exist
        for key in anno_keys:
            s3.download_file(S3_BUCKET_NAME, key, "anno.txt")
            with open("anno.txt", "r") as f:
                lines = f.readlines()
            for line in lines:
                vals = line.strip().split("\t")
                cmd = vals[0].replace("&nbsp ;", "").replace("?", "").strip().lower()
                if cmd in annotated_dict:
                    continue
                else:
                    annotated_dict[cmd] = postprocess_logical_form(
                        cmd, json.loads(vals[1].strip())
                    )

        print(
            f"Size of annotated dict after adding batch-specific annotations: {len(annotated_dict)}"
        )

    # Download meta.txt and compare the three lists of commands from this batch:
    #   - command_dict (from each nsp_outputs.csv)
    #   - collected_commands
    #   - the rows of nsp_data corresponding to the indices in meta.txt
    s3.download_file(S3_BUCKET_NAME, f"{opts.batch_id}/meta.txt", "meta.txt")
    with open("meta.txt", "r") as f:
        meta = f.readlines()
    meta = [int(x.strip()) for x in meta]
    print(f"Length of meta.txt: {len(meta)}")
    anno_cmd_list = [anno_cmd_list[i] for i in range(len(anno_cmd_list)) if i in meta]

    # Remove text_span from annotated dict (some LFs have it and some don't)
    ad_copy = copy.deepcopy(annotated_dict)
    for key in annotated_dict.keys():
        ad_copy[key] = remove_text_span(annotated_dict[key])
    annotated_dict = copy.deepcopy(ad_copy)

    print("\n*** Finished building annotated command dict for batch ***\n")

    return anno_cmd_list, annotated_dict


def main(opts):
    if opts.load_local:
        with open("command_dict.json", "r") as f:
            command_dict = json.load(f)
        with open("error_dict.json", "r") as f:
            error_dict = json.load(f)
        with open("annotated_dict.json", "r") as f:
            annotated_dict = json.load(f)

    else:
        command_dict, error_dict = build_comparison_dicts(opts.batch_id)

        # Check the number of errors against collected_commands
        s3.download_file(
            S3_BUCKET_NAME, f"{opts.batch_id}/collected_commands", "collected_commands.txt"
        )
        with open("collected_commands.txt", "r") as f:
            collected_commands = f.readlines()
        collected_commands = [
            x.replace("&nbsp ;", "").replace("?", "").strip().lower() for x in collected_commands
        ]
        print(
            f"Total number of errors according to `collected_commands` (should match above): {len(collected_commands)}"
        )

        anno_cmd_list, annotated_dict = build_annotated_dict(opts.nsp_data, opts.scrape_anno_outs)

        if opts.save_dicts:
            with open("annotated_dict.json", "w") as f:
                json.dump(annotated_dict, f)
            with open("command_dict.json", "w") as f:
                json.dump(command_dict, f)
            with open("error_dict.json", "w") as f:
                json.dump(error_dict, f)

    # Compare the error v. annotated and command v. annotated dicts
    compare_against_gt(error_dict, annotated_dict, "Labeled NSP errors")
    compare_against_gt(command_dict, annotated_dict, "commands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_id", type=str, default="20220104011348", help="batch ID for interaction job"
    )
    parser.add_argument("--nsp_data", type=str, default="nsp_data_v4.txt")
    parser.add_argument("--save_dicts", action="store_true", default=True)
    parser.add_argument("--load_local", action="store_true", default=False)
    parser.add_argument("--scrape_anno_outs", action="store_true", default=False)
    opts = parser.parse_args()

    main(opts)
