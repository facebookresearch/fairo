"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import ast
import copy
import json
import os
import random
from recombine_data_utils import *
from typing import *


def create_train_valid_split(chunk_index: int, k: int, data_dir: str, output_dir: str):
    """Create partitions for k fold Cross Validation

    Given a chunk index for the valid set, create train and valid split from k chunks of the dataset.
    Chunk index is a an index in the range 0 to k.
    """
    # Read from other chunks and write JSON file to train/ dir
    train_dataset: List[Dict] = []
    valid_dataset: List[Dict] = []
    for i in range(k):
        # Use this as the validation set
        if i == chunk_index:
            valid_dataset += json.load(
                open(data_dir + "cv_pool/chunk_{}/annotated_augmented.json".format(i))
            )
        else:
            train_dataset += json.load(
                open(data_dir + "cv_pool/chunk_{}/annotated_augmented.json".format(i))
            )
    # Write to train and valid directories
    directories: List[str] = ["/", "train/", "valid/"]
    for d in directories:
        if not os.path.isdir(output_dir + d):
            os.mkdir(output_dir + d)

    print(
        "Writing {} entries to {}".format(
            len(train_dataset), output_dir + "train/annotated_augmented.json"
        )
    )
    json.dump(train_dataset, open(output_dir + "train/annotated_augmented.json", "w"))
    print(
        "Writing {} entries to {}".format(
            len(valid_dataset), output_dir + "valid/annotated_augmented.json"
        )
    )
    json.dump(valid_dataset, open(output_dir + "valid/annotated_augmented.json", "w"))


def get_train_annotated_commands(
    data_dir: str, tool1_path: str, tool2_path: str, node_types: List[str]
) -> (List[str], List[str], Dict[str, dict]):
    """
    Fetch Turk data corresponding to annotated data training set.
    """
    # Read from tool 1
    tool1_lines: List[str] = open(tool1_path).readlines()
    # Read from tool 2
    tool2_lines: List[str] = open(tool2_path).readlines()
    # Load the training data that we created
    train_annotated_trees = json.load(open(data_dir + "train/annotated_augmented.json"))
    train_annotated_phrases: List[str] = [x[0] for x in train_annotated_trees]
    turk_processor = TurkToolProcessor(train_annotated_phrases, node_types)
    # Filter samples that we want to use for recombination
    filtered_tool1_lines: List[str] = turk_processor.filter_tool1_lines(tool1_lines)
    filtered_tool2_lines: List[str] = turk_processor.filter_tool2_lines(tool2_lines)
    chat_tree_inserts = turk_processor.build_tree_inserts_dict(tool2_lines)
    return (filtered_tool1_lines, filtered_tool2_lines, chat_tree_inserts)


def create_templates_for_node_type(
    chat: str,
    node_type: str,
    action_dict: dict,
    filtered_tool1_lines: List[str],
    chat_tree_inserts: dict,
) -> (List[tuple], List[tuple]):
    """
    Generate templates and fragments for recombination.
    """
    new_templates = []
    new_fragments = []
    # create recombination template and fragment from tree and chat
    for k, v in ast.literal_eval(action_dict).items():
        if k == node_type:
            if contains_span(v):
                full_tree_with_hole = get_full_tree(
                    chat, node_type, filtered_tool1_lines, chat_tree_inserts
                )
                if full_tree_with_hole is None:
                    print("Error finding the full tree for chat {}".format(chat))
                    break
                span_idxs = get_loc_span_range(full_tree_with_hole, node_type)
                fragment, new_chat = process_chat(chat, span_idxs[1])
                # chat, fragment for subs, original tree with hole (for fragment)
                # original span idxs so we can shift the new ones over
                new_templates.append((new_chat, span_idxs[1], v, full_tree_with_hole))
                # chat fragment, corresponding tree
                new_fragments.append((fragment, span_idxs[1], v))
    return (new_templates, new_fragments)


def gen_chat_tree_templates_and_fragments(
    filtered_tool1_lines, filtered_tool2_lines, chat_tree_inserts, node_types
) -> (Dict[str, list], Dict[str, list]):
    """
    Generate chat and tree fragments and templates.
    """
    full_trees = {}
    fragments = {}

    for l in filtered_tool2_lines:
        chat, child_name, action_dict = l.split("\t")
        if child_name in node_types:
            if child_name not in full_trees:
                full_trees[child_name] = []
            if child_name not in fragments:
                fragments[child_name] = []
            new_templates, new_fragments = create_templates_for_node_type(
                chat, child_name, action_dict, filtered_tool1_lines, chat_tree_inserts
            )
            full_trees[child_name] += new_templates
            fragments[child_name] += new_fragments

    return (full_trees, fragments)


def process_chat(chat: str, span_idxs: list) -> (str, str):
    """Given a chat and span range, remove the span and insert a single <unk> token.
    Return the removed span (Fragment) and processed chat (Template).
    """
    tokens = chat.split(" ")
    fragment = []
    new_tokens = []
    idx = span_idxs[0]
    while idx <= span_idxs[1]:
        fragment.append(tokens[idx])
        idx += 1

    new_tokens += tokens[0 : span_idxs[0]]
    new_tokens.append("<unk>")
    if len(span_idxs) > 1:
        new_tokens += tokens[(span_idxs[1] + 1) : len(tokens)]
    return (" ".join(fragment), " ".join(new_tokens))


def insert_fragment_to_templated_chat(templated_chat: str, fragment: str) -> (str, list):
    """
    Utility for inserting fragments to trees and chats. Note that we deepcopy subtrees.
    """
    chat_str = templated_chat.split(" ")
    new_chat_str = []
    span_idx = []
    for token in chat_str:
        if token == "<unk>":
            span_idx.append(len(new_chat_str))
            new_chat_str += fragment.split(" ")
            span_idx.append(len(new_chat_str) - 1)
        else:
            new_chat_str.append(token)
    return (" ".join(new_chat_str), span_idx)


def insert_subtree_into_full_tree(
    subtree: dict, full_tree: dict, original_span_idx: list, idx_shift: int, span_offset: int
) -> dict:
    """
    Recursively make sure each span node is updated other than the "no"
    """
    new_tree = copy.deepcopy(full_tree)
    for k, v in new_tree.items():
        if type(v) == dict:
            new_tree[k] = insert_subtree_into_full_tree(
                subtree, v, original_span_idx, idx_shift, span_offset
            )

        if type(v) == list:
            if type(v[0]) == str:
                if v[0] == "yes":
                    if type(v[1]) == dict:
                        new_tree[k] = insert_subtree_into_full_tree(
                            subtree, v[1], original_span_idx, idx_shift, span_offset
                        )
                    elif type(v[1]) == list and is_span(v[1]):
                        new_tree[k] = reformat_span_idxs(v[1])
                        if new_tree[k][1][0] > original_span_idx[0]:
                            new_tree[k][1] = [x - idx_shift[1] for x in new_tree[k][1]]
                    else:
                        new_tree[k] = v[1]
                elif v[0] == "no":
                    new_tree[k] = update_tree_spans(copy.deepcopy(subtree), span_offset)
            elif is_span(v):
                new_tree[k] = reformat_span_idxs(v)
                # shift indices over if needed
                if new_tree[k][1][0] > original_span_idx[0]:
                    new_tree[k][1] = [x - idx_shift[1] for x in new_tree[k][1]]
    return new_tree


def update_fragment_tree(tree: dict, offset: int) -> dict:
    """
    Update span positions in a subtree.
    """
    new_tree = copy.deepcopy(tree)
    for key, value in tree.items():
        if type(value) == list and is_span(value):
            reformat_idxs = reformat_span_idxs(value)
            if contains_negative(reformat_idxs[1], offset):
                del new_tree[key]
            else:
                new_tree[key] = [0, [x - offset for x in reformat_idxs[1]]]
        elif type(value) == dict:
            new_tree[key] = update_fragment_tree(value, offset)
    return new_tree


def create_fragment_dataset(subtrees: list, key: str) -> list:
    """
    Creates a dataset of spans given a node type, eg. schematic.
    """
    fragments_dataset = []
    for fragment_set in subtrees:
        text, span, tree = fragment_set
        head = {key: copy.deepcopy(tree)}
        new_tree = postprocess_tree(update_fragment_tree(head, span[0]))
        fragments_dataset.append((text, new_tree["action_sequence"][0]))
    return fragments_dataset


def gen_recombined_data(templates: List[tuple], fragments: List[tuple]) -> List[tuple]:
    """
    Generate recombined examples.
    """
    recombined_data = []
    for i in range(len(templates)):
        for j in range(len(fragments)):
            if i == j:
                continue
            templated_chat, orig_chat_span_idx, templated_tree, templated_full_tree = templates[i]
            fragment, orig_fragment_span_idx, subtree = fragments[j]
            recombined_chat, new_chat_span_idx = insert_fragment_to_templated_chat(
                templated_chat, fragment
            )
            # Calculate shift between original span idx and new span idx
            idx_shift = [
                orig_chat_span_idx[0] - new_chat_span_idx[0],
                orig_chat_span_idx[1] - new_chat_span_idx[1],
            ]
            # span gap for templated chat - orig_chat_span_idx
            # offset for span - orig_fragment_span_idx
            span_offset = orig_fragment_span_idx[0] - new_chat_span_idx[0]
            recombined_full_tree = insert_subtree_into_full_tree(
                subtree, templated_full_tree, orig_chat_span_idx, idx_shift, span_offset
            )
            recombined_full_tree = postprocess_tree(recombined_full_tree)
            recombined_data.append((recombined_chat, recombined_full_tree))
    return recombined_data


def write_recombined_data_chunk(
    data_dir: str,
    output_dir: str,
    tool1_path: str,
    tool2_path: str,
    dataset_name: str,
    node_types: List[str],
    use_fragments: bool,
):
    """
    Read from a partition and write recombined results to output directory.
    """
    filtered_tool1_lines, filtered_tool2_lines, chat_tree_inserts = get_train_annotated_commands(
        data_dir, tool1_path, tool2_path, node_types
    )
    combined_templates, combined_fragments = gen_chat_tree_templates_and_fragments(
        filtered_tool1_lines, filtered_tool2_lines, chat_tree_inserts, node_types
    )
    recombined_data: List[List[str, dict]] = []
    fragments_dataset: List[List[str, dict]] = []

    for key in node_types:
        recombined_data += gen_recombined_data(combined_templates[key], combined_fragments[key])
        fragments_dataset += create_fragment_dataset(combined_fragments[key], key)

    train_output_dir = output_dir + "train/"
    if not os.path.isdir(train_output_dir):
        os.mkdir(train_output_dir)

    if use_fragments:
        random.shuffle(fragments_dataset)
        fragments_data = [[str(x[0]), x[1]] for x in fragments_dataset]
        with open(train_output_dir + dataset_name + "_fragments.json", "w") as outfile:
            print(
                "Writing {} fragments data samples to directory {}".format(
                    len(fragments_data), train_output_dir + dataset_name + ".json"
                )
            )
            json.dump(fragments_dataset, outfile)
    else:
        recombined_data += fragments_dataset
        random.shuffle(recombined_data)
        recombined_data = [[str(x[0]), x[1]] for x in recombined_data]
        print("Created recombined dataset with size {}".format(len(recombined_data)))
        with open(train_output_dir + dataset_name + ".json", "w") as outfile:
            print(
                "Writing {} recombined data samples to directory {}".format(
                    len(recombined_data), train_output_dir + dataset_name + ".json"
                )
            )
            json.dump(recombined_data, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/private/home/rebeccaqian/minecraft/python/craftassist/ttad/data/annotated_data/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--dataset_name",
        default="prompts_recombined_location_ref_objects",
        type=str,
        help="name of recombined dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/checkpoint/rebeccaqian/files/annotated_data/",
        type=str,
        help="directory to write recombined data",
    )
    parser.add_argument(
        "-k", default=10, type=int, help="Number of partitions in leave-k-out-cross-validation."
    )
    parser.add_argument(
        "--create_k_fold_split",
        action="store_true",
        help="Whether to split data into k partitions.",
    )
    parser.add_argument(
        "--fragments", action="store_true", help="Only generate fragments (default is both)."
    )
    parser.add_argument(
        "--node_types",
        default="location,reference_object,schematic",
        type=str,
        help="Comma-separated types of nodes to use for recombination",
    )
    parser.add_argument(
        "--tool1_path",
        default="/private/home/rebeccaqian/minecraft/python/craftassist/text_to_tree_tool/turk_data/tool1/prompts/2_200/all_agreements.txt",
        type=str,
        help="Path to tool1 .txt file",
    )
    parser.add_argument(
        "--tool2_path",
        default="/private/home/rebeccaqian/minecraft/python/craftassist/text_to_tree_tool/turk_data/tool2/prompts/2_200/all_agreements.txt",
        type=str,
        help="Path to tool2 .txt file",
    )
    args = parser.parse_args()
    # types of nodes we want to use for recombination
    node_types = args.node_types.split(",")

    if args.create_k_fold_split:
        for valid_partition_idx in range(args.k):
            output_dir = args.output_dir + "run_{}".format(str(valid_partition_idx)) + "/"
            create_train_valid_split(valid_partition_idx, args.k, args.data_dir, output_dir)
            data_dir = output_dir
            write_recombined_data_chunk(
                data_dir=data_dir,
                output_dir=output_dir,
                tool1_path=args.tool1_path,
                tool2_path=args.tool2_path,
                dataset_name=args.dataset_name,
                node_types=node_types,
                use_fragments=args.fragments,
            )
    else:
        output_dir = args.output_dir
        data_dir = args.data_dir
        write_recombined_data_chunk(
            data_dir=data_dir,
            output_dir=output_dir,
            tool1_path=args.tool1_path,
            tool2_path=args.tool2_path,
            dataset_name=args.dataset_name,
            node_types=node_types,
            use_fragments=args.fragments,
        )


if __name__ == "__main__":
    main()
