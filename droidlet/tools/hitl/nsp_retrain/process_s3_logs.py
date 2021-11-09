"""
This script collects S3 logs from Turk and postprocesses them

Before running this script, ensure that you have a local copy of turk interactions that are in sync with the S3 directory turk_interactions_with_agent.

See README for instructions.
"""

import glob
import pandas as pd
import re
from datetime import datetime
import argparse
import boto3
import os
import tarfile
import re

pd.set_option("display.max_rows", 10)


def read_s3_bucket(s3_logs_dir, output_dir):
    print(
        "{s3_logs_dir}/**/{csv_filename}".format(
            s3_logs_dir=s3_logs_dir, csv_filename="logs.tar.gz"
        )
    )
    # NOTE: This assumes the local directory is synced with the same name as the S3 directory
    pattern = re.compile(r".*turk_logs/(.*)/logs.tar.gz")
    # NOTE: this is hard coded to search 2 levels deep because of how our logs are structured
    for csv_path in glob.glob(
        "{s3_logs_dir}/**/{csv_filename}".format(
            s3_logs_dir=s3_logs_dir, csv_filename="logs.tar.gz"
        )
    ):
        tf = tarfile.open(csv_path)
        timestamp = pattern.match(csv_path).group(1)
        tf.extractall(path="{}/{}/".format(output_dir, timestamp))

def get_stats(command_list):
    AC = ["build", "move", "destroy", "dance", "get", "tag", "dig", "copy", "undo", "fill", "spawn", "answer", "stop", "resume", "come", "go"]
    command_list = [c.lower() for c in command_list]
    len_ori = len(command_list)
    l = list(set(command_list))
    len_dedup = len(l)

    total_len = 0
    interested = 0
    for c in l:
        # print(c)
        if any(word in c.lower() for word in AC):
            interested += 1
        total_len += len(c.split())
    avg_len = total_len / len_dedup

    print(f'num_ori {len_ori}')
    print(f'num_dedup {len_dedup}')
    print(f'dup_rate {(len_ori - len_dedup) / len_ori * 100}%')
    print(f'avg_len {avg_len}')
    print(f'valid {interested}')
    print(f'valid rate {interested / len_dedup * 100}%')

def read_turk_logs(turk_output_directory, filename):
    # Crawl turk logs directory
    all_turk_interactions = None
    for csv_path in glob.glob(
        "{turk_logs_dir}/**/{csv_filename}".format(
            turk_logs_dir=turk_output_directory, csv_filename=filename + ".csv"
        )
    ):
        with open(csv_path) as fd:
            # collect the NSP outputs CSV
            csv_file = pd.read_csv(csv_path, delimiter="|")
            # add a column with the interaction log ID
            interaction_log_id = re.search(r"\/([^\/]*)\/{}.csv".format(filename), csv_path).group(
                1
            )
            csv_file["turk_log_id"] = interaction_log_id
            if all_turk_interactions is None:
                all_turk_interactions = csv_file
            else:
                all_turk_interactions = pd.concat(
                    [all_turk_interactions, csv_file], ignore_index=True
                )

    if all_turk_interactions is None:
        return []

    # Drop duplicates
    all_turk_interactions.drop_duplicates()
    # print(all_turk_interactions.head())
    # print(all_turk_interactions.shape)

    get_stats(list(all_turk_interactions["command"]))
    # return all commands as a list
    return list(set(all_turk_interactions["command"]))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # User needs to provide file I/O paths
    # parser.add_argument(
    #     "--turk_logs_directory",
    #     help="where to read s3 logs from eg. ~/turk_interactions_with_agent",
    # )
    # parser.add_argument(
    #     "--parsed_output_directory",
    #     help="where to write the collated NSP outputs eg. ~/parsed_turk_logs",
    # )
    # parser.add_argument(
    #     "--filename",
    #     default="nsp_outputs",
    #     help="name of the CSV file we want to read, eg. nsp_outputs",
    # )
    # args = parser.parse_args()
    # read_s3_bucket(args.turk_logs_directory, args.parsed_output_directory)
    # read_turk_logs(args.parsed_output_directory, args.filename)

    # read_s3_bucket("/private/home/yuxuans/.tmp/turk_logs", "/private/home/yuxuans/.tmp/parsed")
    NSP_OUTPUT_FNAME = "nsp_outputs"
    print('-------NSP---------')
    read_turk_logs("/private/home/yuxuans/.tmp/parsed", NSP_OUTPUT_FNAME)
    print('-------error---------')
    NSP_OUTPUT_FNAME = "error_details"
    read_turk_logs("/private/home/yuxuans/.tmp/parsed", NSP_OUTPUT_FNAME)
