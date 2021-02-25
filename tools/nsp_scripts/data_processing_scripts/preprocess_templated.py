"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
from random import shuffle

def preprocess(input_path, output_path):
    """Loads a templated dataset file and processes it into format read for training.
    - Discard examples with multiple chats
    - Append delimiters and remove blank lines

    Args:
        input_path: input .txt file
        output_path: location to write output dataset

    Format of dataset: [text]|[logical_form]
    """
    with open(input_path, "r") as f:
        examples = []
        commands = []
        for line in f:
            if line.strip() == '':
                if len(commands) > 2:
                    commands = []
                    continue
                examples.append("|".join(commands))
                commands = []
            else:
                commands += [line.strip()]
        shuffle(examples)

    # write examples to ground truth file
    with open(output_path, "w") as fd:
        for line in examples:
            fd.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="generated_dialogues.txt")
    parser.add_argument("--output_path", type=str, default="craftassist/agent/datasets/full_data/templated.txt")
    args = parser.parse_args()
    preprocess(args.raw_data_path, args.output_path)