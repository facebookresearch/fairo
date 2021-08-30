"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse


def print_csv_format(filename, option_num, max_words=40):
    if option_num == 1:
        # level 1, tool A
        print("command", *["word{}".format(i) for i in range(max_words)], sep=",")

        with open(filename) as f:
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This is default option for plain text to be rendered.
                words = command.split()
                print(command, *words, *([""] * (max_words - len(words))), sep=",")

    elif option_num == 2:
        # level 2, tool B
        print(
            "command",
            "intent",
            "child",
            "highlight_words",
            *["word{}".format(i) for i in range(max_words)],
            sep=","
        )

        with open(filename) as f:
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This option is if we need highlighted text to be rendered
                # file will have : text + "\t" +  text with spans in for highlighted words + "\t" + action_name + "\t" + child_name
                parts = command.split("\t")
                words = parts[0].split()
                intent = parts[2]
                child = parts[3]
                highlight_words = parts[4].replace(",", "-")
                print(
                    parts[1],
                    intent,
                    child,
                    highlight_words,
                    *words,
                    *([""] * (max_words - len(words))),
                    sep=","
                )
    elif option_num == 3:
        # tool C (for reference object only)
        print(
            "command",
            "intent",
            "child",
            "highlight_words",
            *["word{}".format(i) for i in range(max_words)],
            sep=","
        )

        with open(filename) as f:
            for line in f.readlines():
                command = line.strip()
                # This option is if we need highlighted text to be rendered
                # file will have :
                # text + "\t" +  text with spans in for highlighted words + "\t" + action_name + "\t" + reference_object
                """NOTE: if location and filters allowed as reference_object children: + "\t" + ref_object_child_name"""
                parts = command.split("\t")
                words = parts[0].replace(",", "").split()  # all words for spans
                intent = parts[2].replace(",", "")
                child = parts[3].replace(",", "")
                highlight_words = parts[4].replace(",", "-")
                print(
                    parts[1],
                    intent,
                    child,
                    highlight_words,
                    *words,
                    *([""] * (max_words - len(words))),
                    sep=","
                )
    elif option_num == 4:
        # tool D, comparison for filters tool
        print(
            "command",
            "child",
            "ref_child",
            "highlight_words",
            *["word{}".format(i) for i in range(max_words)],
            sep=","
        )

        with open(filename) as f:
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This option is if we need highlighted text to be rendered
                # file will have :
                # text + "\t" +  text with spans in for highlighted words + "\t" + reference_object + "\t" + "comparison"
                parts = command.split("\t")
                words = parts[0].split()  # all words for spans
                child = parts[2]
                ref_child = parts[3]
                highlight_words = parts[4].replace(",", "-")
                print(
                    parts[1],
                    child,
                    ref_child,
                    highlight_words,
                    *words,
                    *([""] * (max_words - len(words))),
                    sep=","
                )
    elif option_num == 5:
        # qualification test
        print(
            "command_1",
            *["word1{}".format(i) for i in range(max_words)],
            "command_2",
            *["word2{}".format(i) for i in range(max_words)],
            "command_3",
            *["word3{}".format(i) for i in range(max_words)],
            sep=","
        )

        with open(filename) as f:
            l = []
            for line in f.readlines():
                command = line.replace(",", "").strip()
                # This is default option for plain text to be rendered.
                words = command.split()
                l.append(",".join([command, *words, *([""] * (max_words - len(words)))]))
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
    parser.add_argument("--max_words", type=int, default=40)

    args = parser.parse_args()

    print_csv_format(args.input_file, args.tool_num, args.max_words)
