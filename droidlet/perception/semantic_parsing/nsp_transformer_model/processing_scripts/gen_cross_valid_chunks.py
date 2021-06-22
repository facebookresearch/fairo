"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import os
import random
import math
from typing import *


def partition_dataset(k: int, data_path: str, output_dir: str):
    """
    Split dataset into k partitions.
    """
    # Read in annotated dataset
    full_dataset = open(data_path + "annotated_data_text_spans.txt").readlines()
    print("Length of dataset: {} lines".format(len(full_dataset)))

    # shuffle
    random.shuffle(full_dataset)
    chunk_size = int(math.ceil(len(full_dataset) / float(k)))
    print("k is {}".format(k))
    print("Chunk size: {}".format(chunk_size))

    start_idx = 0
    # Split into k chunks
    for i in range(k):
        end_idx = start_idx + chunk_size
        chunk_i = full_dataset[start_idx : min(end_idx, len(full_dataset))]
        print("Length of chunk {}: {}".format(i, len(chunk_i)))

        partition_path = output_dir + "chunk_{}/".format(i)
        if not os.path.isdir(partition_path):
            os.mkdir(partition_path)
        with open(partition_path + "annotated.txt", "w") as fd:
            for line in chunk_i:
                fd.write(line)
        start_idx += chunk_size


def create_train_valid_split(chunk_index: int, k: int, data_dir: str, output_dir: str):
    """Create partitions for k fold Cross Validation

    Given a chunk index for the valid set, create train and valid split from k chunks of the dataset.
    Chunk index is a an index in the range 0 to k.
    """
    # Read from other chunks and write txt file to train/ dir
    train_dataset: List[Dict] = []
    valid_dataset: List[Dict] = []
    for i in range(k):
        # Use this as the validation set
        if i == chunk_index:
            valid_dataset += open(data_dir + "chunk_{}/annotated.txt".format(i)).readlines()
        else:
            train_dataset += open(data_dir + "chunk_{}/annotated.txt".format(i)).readlines()
    # Write to train and valid directories
    directories: List[str] = ["/", "train/", "valid/"]
    for d in directories:
        if not os.path.isdir(output_dir + d):
            os.mkdir(output_dir + d)

    print(
        "Writing {} entries to {}".format(len(train_dataset), output_dir + "train/annotated.txt")
    )
    with open(output_dir + "train/annotated.txt", "w") as fd:
        for line in train_dataset:
            fd.write(line)

    print(
        "Writing {} entries to {}".format(len(valid_dataset), output_dir + "valid/annotated.txt")
    )
    with open(output_dir + "valid/annotated.txt", "w") as fd:
        for line in valid_dataset:
            fd.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/checkpoint/rebeccaqian/datasets/06_24/",
        type=str,
        help="Contains the full dataset to partition",
    )
    parser.add_argument(
        "--output_dir",
        default="/checkpoint/rebeccaqian/datasets/06_24/",
        type=str,
        help="Directory to write data partitions for CV runs",
    )
    parser.add_argument(
        "-k", default=10, type=int, help="Number of partitions in leave-k-out-cross-validation."
    )
    args = parser.parse_args()
    partition_dataset(args.k, args.data_dir, args.output_dir)

    for valid_partition_idx in range(args.k):
        chunk_path = args.output_dir + "run_{}/".format(valid_partition_idx)
        if not os.path.isdir(chunk_path):
            os.mkdir(chunk_path)
        output_dir = chunk_path
        create_train_valid_split(valid_partition_idx, args.k, args.data_dir, output_dir)


if __name__ == "__main__":
    main()
