"""
Remove static validation commands from the dataset
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        default="/private/home/rebeccaqian/minecraft/base_agent/ttad/generation_dialogues/templated_pre.txt"
    )
    parser.add_argument(
        "--commands_to_remove",
        help="path to test set commands to remove",
        default="/private/home/rebeccaqian/minecraft/base_agent/ttad/generation_dialogues/templated_pre.txt"
    )
    parser.add_argument(
        "--output_path",
        help="path of cleaned dataset to be used in train/valid split",
        default="/private/home/rebeccaqian/minecraft/base_agent/ttad/generation_dialogues/templated_pre.txt"
    )
    opts = parser.parse_args()
    with open(opts.commands_to_remove) as fd:
        commands_to_remove = fd.read().splitlines()

    cleaned_dataset = []
    with open(opts.data_source) as fd:
        dataset = fd.readlines()
        for line in dataset:
            command, parse_tree = line.split("|")
            if command in commands_to_remove:
                continue
            else:
                cleaned_dataset.append(line)

    # Write cleaned data
    with open(opts.output_path, "w") as fd:
        for row in cleaned_dataset:
            fd.write(row)
