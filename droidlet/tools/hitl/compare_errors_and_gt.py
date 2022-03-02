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
from typing import List
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


def main(opts):
    if opts.load_local:
        with open("error_dict.json", "r") as f:
            error_dict = json.load(f)
        with open("annotated_dict.json", "r") as f:
            annotated_dict = json.load(f)

    else:
        # Retrieve all of the log file keys from S3 bucket for the given batch_id
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{opts.batch_id}/interaction")
        log_keys = [item["Key"] for item in response["Contents"]]

        # Download each log file, extract, and pull the errors into a dictionary
        error_dict = {}
        for key in log_keys:
            s3.download_file(S3_BUCKET_NAME, key, "logs.tar.gz")
            tf = tarfile.open("logs.tar.gz")
            tf.extract("./error_details.csv")
            csv_file = pd.read_csv("error_details.csv", delimiter="|")
            for idx, row in csv_file.iterrows():
                if row["command"] in error_dict:
                    error_dict[row["command"]].append(
                        json.loads(row["action_dict"].replace("'", '"'))
                    )
                else:
                    error_dict[row["command"]] = [json.loads(row["action_dict"].replace("'", '"'))]

        # Remove text_span from error dict
        ed_copy = copy.deepcopy(error_dict)
        for key in error_dict.keys():
            for i in range(len(error_dict[key])):
                ed_copy[key][i] = remove_text_span(error_dict[key][i])
        error_dict = copy.deepcopy(ed_copy)

        # Save to be loaded later
        with open("error_dict.json", "w") as f:
            json.dump(error_dict, f)

        # Download nsp_data and build a annotated command dict
        s3.download_file(S3_BUCKET_NAME, opts.nsp_data, "nsp_data.txt")
        with open("nsp_data.txt", "r") as f:
            nsp_data = f.readlines()
        annotated_dict = {}
        for line in nsp_data:
            split_line = line.strip().split("|")
            try:
                if len(split_line) == 2:
                    annotated_dict[split_line[0].strip()] = postprocess_logical_form(
                        split_line[0].strip(), json.loads(split_line[1].strip())
                    )
                else:
                    annotated_dict[split_line[1].strip()] = postprocess_logical_form(
                        split_line[1].strip(), json.loads(split_line[2].strip())
                    )
            except:
                print(split_line)
                raise

        # Retrieve all of the annotated command file keys from S3 bucket for the given batch_id
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{opts.batch_id}/annotated")
        anno_keys = []
        for page in pages:
            anno_keys += [item["Key"] for item in page["Contents"]]

        print(f"Size of error dict: {len(error_dict)}")
        print(f"Number of annotated commands (keys): {len(anno_keys)}")
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
                if vals[0] in annotated_dict:
                    continue
                else:
                    annotated_dict[vals[0].strip()] = postprocess_logical_form(
                        vals[0].strip(), json.loads(vals[1].strip())
                    )

        print(
            f"Size of annotated dict after adding batch-specific annotations: {len(annotated_dict)}"
        )

        # Remove text_span from annotated dict
        ad_copy = copy.deepcopy(annotated_dict)
        for key in annotated_dict.keys():
            ad_copy[key] = remove_text_span(annotated_dict[key])
        annotated_dict = copy.deepcopy(ad_copy)

        # Save constructed dicts locally to save time on previous runs
        with open("annotated_dict.json", "w") as f:
            json.dump(annotated_dict, f)

    # Compare the two dicts and report results
    correct_errors = 0
    total_errors = 0
    not_found = 0
    for key in error_dict.keys():
        for ad in error_dict[key]:
            if key not in annotated_dict:
                not_found += 1
                continue
            if ad != annotated_dict[key]:
                correct_errors += 1
            total_errors += 1

    print(f"# no annotated GT found: {not_found}")
    print(f"Errors labeled correctly: {correct_errors}")
    print(f"Incorrect errors (model parse was right): {total_errors - correct_errors}")
    print(f"Total Errors Labeled: {total_errors + not_found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_id", type=str, default="20220104011348", help="batch ID for interaction job"
    )
    parser.add_argument("--nsp_data", type=str, default="nsp_data_v4.txt")
    parser.add_argument("--load_local", action="store_true", default=False)
    opts = parser.parse_args()

    main(opts)
