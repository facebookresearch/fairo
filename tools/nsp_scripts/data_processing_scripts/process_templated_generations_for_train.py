import json
import argparse
from random import shuffle, seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_templated_path",
        default="/private/home/rebeccaqian/minecraft/base_agent/ttad/generation_dialogues/templated_pre.txt",
    )
    parser.add_argument(
        "--ground_truth_path",
        default="/private/home/rebeccaqian/minecraft/craftassist/agent/datasets/ground_truth/datasets/templated.txt",
    )
    parser.add_argument(
        "--templated_path",
        default="/private/home/rebeccaqian/minecraft/base_agent/ttad/generation_dialogues/templated_pre.txt",
    )
    parser.add_argument(
        "--full_path",
        default="/private/home/rebeccaqian/minecraft/craftassist/agent/datasets/full_data/templated.txt",
    )
    parser.add_argument(
        "--train_dir_path",
        default="/private/home/rebeccaqian/minecraft/craftassist/agent/datasets/annotated_data/",
    )
    opts = parser.parse_args()
    f = open(opts.raw_templated_path)
    examples = []
    commands = []
    for line in f:
        if line.strip() == "":
            if len(commands) > 2:
                commands = []
                continue
            examples.append("|".join(commands))
            commands = []
        else:
            commands += [line.strip()]

    f.close()

    train_idx = int(round(len(examples) * 0.8))
    # test_idx = train_idx + int(round(len(examples) * 0.2))
    train = examples[:train_idx]
    # test = examples[train_idx:test_idx]
    valid = examples[train_idx : len(examples)]
    print(train[0])

    # write examples to ground truth file
    with open(opts.ground_truth_path, "w") as fd:
        for line in examples:
            fd.write(line + "\n")

    with open(opts.full_path, "w") as fd:
        for line in examples:
            fd.write(line + "\n")

    with open(opts.train_dir_path + "train/templated.txt", "w") as fd:
        for line in train:
            fd.write(line + "\n")

    with open(opts.train_dir_path + "valid/templated.txt", "w") as fd:
        for line in valid:
            fd.write(line + "\n")
