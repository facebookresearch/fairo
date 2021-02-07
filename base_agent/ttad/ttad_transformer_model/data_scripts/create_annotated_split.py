"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import ast
import argparse
import math
from os import mkdir
from os.path import isdir

from random import shuffle, seed


def create_data_split(data_path, filename, output_path, split_ratios):
    """Splits a dataset into train/test/valid splits.

    Args:
        data_path (str): directory to load from
        filename (str): name of file to load
        output_path (str): directory to write the output files
        split_ratios (dict): ratio of each data split

    """
    annotated_data = open(data_path + filename).readlines()
    shuffle(annotated_data)
    start_idx = 0
    for split_name, ratio in split_ratios:
        num_samples = int(round(len(annotated_data) * ratio))
        end_idx = start_idx + num_samples
        chunk = annotated_data[start_idx : min(end_idx, len(annotated_data))]
        print(chunk[0])
        write_data_split(output_path, "{}/".format(split_name), filename, chunk)
        start_idx = end_idx


def write_data_split(output_path, split, filename, data_chunk):
    """Writes a chunk of data to the output file path."""
    dirpath = output_path + "/" + split
    filepath = dirpath + filename
    if not isdir(dirpath):
        mkdir(dirpath)
    with open(filepath, "w") as fd:
        for line in data_chunk:
            fd.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path", type=str, default="craftassist/agent/datasets/full_data/"
    )
    parser.add_argument(
        "--output_path", type=str, default="craftassist/agent/datasets/annotated_data/"
    )
    parser.add_argument("--filename", type=str, default="annotated.txt")
    parser.add_argument(
        "--split_ratio", type=str, default="0.7:0.2:0.1", help="train:test:valid split of dataset"
    )
    args = parser.parse_args()

    split_names = ["train", "test", "valid"]
    split_ratios = zip(split_names, [float(x) for x in args.split_ratio.split(":")])

    create_data_split(args.raw_data_path, args.filename, args.output_path, split_ratios)
