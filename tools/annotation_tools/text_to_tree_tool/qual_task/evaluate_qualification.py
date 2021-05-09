"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import Counter, defaultdict
import argparse
import ast

right_answer_count = Counter()
wrong_answer_count = Counter()

# compile sets of allowed answers
allowed_answers = defaultdict(set)
command = None


def read_gold_set(gold_set):
    command = None
    with open(gold_set, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line.startswith("{"):
                try:
                    allowed_answers[command].add(line)
                except:
                    print("Bad allowed answer:", line)
                    raise
            else:
                command = line


def compare_dicts(action_dict, allowed_dict):
    # action_dict = ast.literal_eval(action_dict)
    allowed_dict = ast.literal_eval(allowed_dict)
    if "repeat" in allowed_dict:
        if "repeat" not in action_dict:

            return False
        val = allowed_dict["repeat"]
        val2 = action_dict["repeat"]
        if val != val2:
            if val[0] != val2[0]:
                return False
            val_dict1 = val[1]
            val_dict2 = val2[1]
            for k, v in val_dict2.items():
                if k == "repeat_dir":
                    continue
                if k not in val_dict1 or v != val_dict1[k]:
                    return False

    for k, v in allowed_dict.items():
        if k == "repeat":
            continue
        if k not in action_dict or action_dict[k] != v:
            return False
    return True


def get_wrong_stats(dict1, dict2):
    """{'repeat',
    'schematic',
    'dialogue_type',
    'action_type',
    'has_block_type',
    'reference_object',
    'tag_val',
    'filters',
    'location',
    'target_action_type'}"""
    st = {}
    for k, v in dict2:
        if k not in dict1:
            print("missing key: %r" % (k))
        if v != dict1[k]:
            st[k] += 1


def evaluate_workers(worker_file):
    worker_stats = {}
    wrong_stats = {}
    for k, v in allowed_answers.items():
        wrong_stats[k] = {}
    with open(worker_file) as f:
        for line in f.readlines():
            right_count = 0
            wrong_count = 0
            worker_id, answers = line.strip().split("\t")
            answer_dicts = ast.literal_eval(answers)

            # if worker didn't answer all questions, ignore
            if len(answer_dicts.keys()) < 3:
                print("Skipping: %r completed only %r" % (worker_id, len(answer_dicts.keys())))
                continue

            # otherwise read all answers
            # k is sentence, v is dict
            for k, v in answer_dicts.items():
                # k has to be in allowed_answers
                if k not in allowed_answers:
                    print("BADDDDDD")

                # if answer doesn't match any allowed answer
                if not any(compare_dicts(v, d) for d in allowed_answers[k]):
                    wrong_count += 1
                    for d in allowed_answers[k]:
                        stats = get_wrong_stats(v, d)
                        wrong_stats[k].update(stats)
                        return worker_stats
                    print(k, v)
                else:
                    right_count += 1
            # print("-" * 30)
            worker_stats[worker_id] = int((right_count / (right_count + wrong_count)) * 100)

    return worker_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_set", type=str, required=True)
    parser.add_argument("--worker_file", type=str, required=True)
    parser.add_argument("--worker_stats_out", type=str, required=True)
    args = parser.parse_args()

    read_gold_set(args.gold_set)
    stats = evaluate_workers(args.worker_file)

    with open(args.worker_stats_out, "w") as f:
        for worker_id, val in stats.items():
            f.write(worker_id + "\t" + str(val) + "\n")
