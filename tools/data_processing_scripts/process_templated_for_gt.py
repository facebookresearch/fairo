"""
This script processes large files eg. templated generations to create ground truth annotations.
"""
import argparse


def remove_duplicates(input_path, output_path):
    """Remove duplicate commands in dataset.
    """
    new_data = {}
    with open(input_path) as fd:
        data = fd.readlines()

    for line in data:
        command, action_dict = line.split("|")
        words_arr = command.split(" ")
        if command in new_data:
            continue
        # Skip long comomands and NOOPs
        elif len(words_arr) > 6 or "NOOP" in line:
            continue
        else:
            new_data[command] = action_dict

    print(len(new_data))

    with open(output_path, "w") as fd:
        for command in new_data:
            fd.write("{}|{}".format(command, new_data[command]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="File we want to process for use in ground truth annotations.",
        # Assuming run from ~/droidlet
        default="craftassist/agent/datasets/full_data/templated.txt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where to write groound truth annotations to.",
        default="craftassist/agent/datasets/ground_truth/datasets/templated.txt"
    )
    args = parser.parse_args()
    remove_duplicates(args.input_path, args.output_path)

