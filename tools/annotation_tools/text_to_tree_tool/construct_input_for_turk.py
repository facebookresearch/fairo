"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse

from annotation_tool_1 import MAX_WORDS


def print_csv_format(filename, option_num):
    if option_num == 1:
        # level 1
        print("command", *["word{}".format(i) for i in range(MAX_WORDS)], sep=",")

        with open(filename) as f:
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This is default option for plain text to be rendered.
                words = command.split()
                print(command, *words, *([""] * (MAX_WORDS - len(words))), sep=",")

    elif option_num == 2:
        # level 2
        print(
            "command", "intent", "child", *["word{}".format(i) for i in range(MAX_WORDS)], sep=","
        )

        with open(filename) as f:
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This option is if we need highlighted text to be rendered
                # file will have : text + "\t" +  text with spans in for highlighted words
                parts = command.split("\t")
                words = parts[0].split()
                intent = parts[2]
                child = parts[3]
                print(parts[1], intent, child, *words, *([""] * (MAX_WORDS - len(words))), sep=",")
    elif option_num == 3:
        # qualification test
        print(
            "command_1",
            *["word1{}".format(i) for i in range(MAX_WORDS)],
            "command_2",
            *["word2{}".format(i) for i in range(MAX_WORDS)],
            "command_3",
            *["word3{}".format(i) for i in range(MAX_WORDS)],
            sep=","
        )

        with open(filename) as f:
            l = []
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This is default option for plain text to be rendered.
                words = command.split()
                l.append(",".join([command, *words, *([""] * (MAX_WORDS - len(words)))]))
            print(",".join(l))
    elif option_num == 4:
        # composite command tool
        print("sentence")

        with open(filename) as f:
            for line in f.readlines():
                line = line.strip()
                print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--tool_num", type=int, default=1)

    args = parser.parse_args()

    print_csv_format(args.input_file, args.tool_num)
