"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import ast
import copy
from enum import Enum
from typing import *


class TurkToolProcessor:
    """Methods for fetching turk tool outputs and various utils for data recombination.

    Attributes:
        train_annotated_phrases (List[str]): all chat commands in training data
        node_types (List[str]): nodes we want to swap fragments for
    """

    def __init__(self, train_annotated_phrases: List[str], node_types: List[str]):
        self.train_annotated_phrases = train_annotated_phrases
        self.node_types = node_types

    def filter_tool1_lines(self, tool1_lines: List[str]) -> List[str]:
        """Fetch Turk tool1 lines that correspond to the training dataset.
        -- tool1_lines: input from Turk tool, with format <text> \t <action dict>
        -- node_types: list of nodes we want to swap, eg. location
        Source: https://github.com/fairinternal/minecraft/tree/master/python/craftassist/text_to_tree_tool/turk_data/tool1/
        """
        filtered_lines = []
        for l in tool1_lines:
            text, action_dict = l.split("\t")
            if text in self.train_annotated_phrases:
                # Check that the tree can be expanded
                if "no" not in action_dict:
                    continue
                for key in self.node_types:
                    if key in action_dict:
                        filtered_lines.append(l)
                        break
        return filtered_lines

    def is_chat_subset(self, chat: str) -> bool:
        """
        Checks whether a chat fits criteria for recombination.
        """
        return (
            chat in self.train_annotated_phrases
            and len(chat.split(" ")) <= 30
            and "composite_action" not in chat
        )

    def filter_tool2_lines(self, tool2_lines: List[str]) -> List[str]:
        """Fetch Turk tool2 lines that correspond to the training dataset.
        
        tool2_lines -- input from Turk tool, with format <chat> \t <tag> \t <action dict>
        node_types -- list of nodes we want to swap, eg. location
        Source: https://github.com/fairinternal/minecraft/tree/master/python/craftassist/text_to_tree_tool/turk_data/tool2/
        """
        filtered_lines = []
        for l in tool2_lines:
            chat, tag, action_dict = l.split("\t")
            use_chat = self.is_chat_subset(chat)
            if tag in self.node_types and use_chat:
                filtered_lines.append(l)
        return filtered_lines

    def build_tree_inserts_dict(self, lines: List[str]) -> Dict[str, dict]:
        """
        Build tree representation of chats and all inserts for the full tree
        """
        chat_tree_inserts = {}
        for l in lines:
            chat, tag, action_dict = l.split("\t")
            if chat in self.train_annotated_phrases:
                if chat not in chat_tree_inserts:
                    chat_tree_inserts[chat] = {tag: ast.literal_eval(action_dict)}
                else:
                    chat_tree_inserts[chat][tag] = ast.literal_eval(action_dict)
        return chat_tree_inserts


def get_full_tree(
    chat: str, key: str, tool1_lines: List[str], chat_tree_inserts: List[dict]
) -> Union[dict, None]:
    """
    Given a chat command, fetch the full tree minus the node we want to swap.
    """
    # Fetch the corresponding full tree with hole
    for full_line in tool1_lines:
        if chat in full_line:
            _, full_tree = full_line.split("\t")
            full_tree_dict = ast.literal_eval(full_tree)
            # fill other holes
            full_tree_dict = fill_other_holes(key, full_tree_dict, chat, chat_tree_inserts)
            return full_tree_dict
    return None


def fill_other_holes(key: str, full_tree: dict, chat: str, chat_tree_inserts: dict) -> dict:
    """Given a full tree (with holes), node type of swaps and the chat command, fill the holes for nodes
    that we are not swapping.
    """
    new_tree = copy.deepcopy(full_tree)
    for k, v in new_tree.items():
        if v[0] == "no" and k != key:
            if type(v[1]) == list:
                new_tree[k] = v[1]
            # get the subtree from the grouped document by the chat instruction
            tree_insert = copy.deepcopy(chat_tree_inserts[chat][k][k])
            new_tree[k] = tree_insert
    return new_tree


class SpanType(Enum):
    TURK = 1
    DATASET = 2


def is_span(val: Any) -> Union[bool, SpanType]:
    """
    Check if a value is a span type.
    """
    if type(val) == list:
        # Unformatted spans
        if type(val[0]) == list and type(val[0][0]) == int:
            return SpanType.TURK
        # Formatted spans
        if type(val[0]) == int and type(val[1]) == list:
            return SpanType.DATASET
    return False


def contains_negative(span: List[int], offset: int) -> bool:
    """
    Checks if a recombined span contains negative values.
    """
    for idx in span:
        if idx - int(offset) < 0:
            return True
    return False


def contains_span(tree: dict) -> bool:
    """
    Whether a tree contains span nodes.
    """
    for k, v in tree.items():
        if type(v) == dict:
            return contains_span(v)
        if type(v) == list:
            if is_span(v):
                return True
    return False


def get_loc_span_range(tree: dict, key: str) -> Union[list, None]:
    """
    Fetch the span range for the subtree to be inserted into the full tree (with hole).
    """
    for k, v in tree.items():
        if k == key:
            if type(v) == list and v[0] == "no":
                span_range = reformat_span_idxs(v[1])
                return span_range
    return None


def reformat_span_idxs(span_idx_list: List[List[int]]) -> list:
    """
    Reformat span idxs to look like [0, [start_idx, end_idx]]
    """
    span_idxs = [x[0] for x in span_idx_list]

    start_idx = min(span_idxs)
    end_idx = max(span_idxs)
    return [0, [start_idx, end_idx]]


def update_tree_spans(tree: dict, shift: int) -> dict:
    """Insert recombined tree into parent tree. 
    Get rid of "yes" and "no" indicators, reformat tree, insert subtrees.
    """
    new_tree = copy.deepcopy(tree)
    for k, v in new_tree.items():
        if type(v) == dict:
            new_tree[k] = update_tree_spans(v, shift)
        span_type = is_span(v)
        if span_type:
            if span_type == SpanType.TURK:
                new_tree[k] = reformat_span_idxs(v)
            new_tree[k][1] = [x - shift for x in new_tree[k][1]]
    return new_tree


def postprocess_tree(tree: dict) -> dict:
    """
    Process a turk tool generated tree to fit the format for model datasets.
    """
    new_tree = copy.deepcopy(tree)
    new_tree = postprocess_tree_helper(new_tree)
    new_tree["action_sequence"] = [{}]
    for k, v in copy.deepcopy(new_tree).items():
        if k != "dialogue_type" and k != "action_sequence":
            for i in range(len(new_tree["action_sequence"])):
                action_dict = new_tree["action_sequence"][i]
                if k not in action_dict:
                    action_dict[k] = v
                elif i == len(new_tree["action_sequence"]) - 1:
                    new_tree["action_sequence"].append({k: v})

            del new_tree[k]

    return new_tree


def postprocess_tree_helper(tree: dict) -> dict:
    """
    Post processing recombined tree.
    """
    new_tree = copy.deepcopy(tree)
    for k, v in tree.items():
        if k == "action_type":
            if v == "copy":
                new_tree[k] = "build"
            new_tree[k] = new_tree[k].upper()
        if k == "contains_coreference" and v == "no":
            del new_tree[k]
        if type(v) == dict:
            new_tree[k] = postprocess_tree_helper(v)
    return new_tree
