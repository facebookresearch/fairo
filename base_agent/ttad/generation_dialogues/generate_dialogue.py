"""
Copyright (c) Facebook, Inc. and its affiliates.

This file generates action trees and language based on options from
command line.
"""
import json
from generate_data import *


class Action(ActionNode):
    """options for Actions"""

    CHOICES = [
        Move,
        Build,
        Destroy,
        Noop,
        Stop,
        Resume,
        Dig,
        Copy,
        Undo,
        Fill,
        Spawn,
        Freebuild,
        Dance,
        GetMemory,
        PutMemory,
    ]


human_give_command_actions = [
    Move,
    Build,
    Destroy,
    Stop,
    Resume,
    Dig,
    Copy,
    Undo,
    Fill,
    Spawn,
    Freebuild,
    Dance,
]


# Mapping of command line action type to action class
action_type_map = {
    "move": Move,
    "build": Build,
    "destroy": Destroy,
    "noop": Noop,
    "stop": Stop,
    "resume": Resume,
    "dig": Dig,
    "copy": Copy,
    "undo": Undo,
    "fill": Fill,
    "spawn": Spawn,
    "freebuild": Freebuild,
    "dance": Dance,
    "get_memory": GetMemory,
    "put_memory": PutMemory,
}

arg_numeric_range_map = {
    "length_range": "length",
    "width_range": "width",
    "depth_range": "depth",
    "height_range": "height",
    "size_range": "size",
    "thickness_range": "thickness",
    "radius_range": "radius",
    "slope_range": "slope",
    "distance_range": "distance",
    "step_range": "step",
    "count_range": "count",
    "base_range": "base",
    "coordinates_range": "coordinates",
}

arg_name_file_map = {
    "non_shape_names": "non_shape_names",
    "block_types_file": "block_types",
    "mob_file": "mob_names",
}

child_key_map = {"BUILD": ["schematic", "reference_object"], "DESTROY": ["reference_object"]}


def add_new_similar_action(action_text, previous_action, name_2, curr_action, curr_index):
    """Append curr_action's action dict to previous_action's action_sequenceself.
    curr_index is index of current action dict in action action_sequence"""

    span = find_span(action_text, name_2)
    action_type = curr_action["action_type"]
    child_key = child_key_map[action_type]
    child_key_name = None
    for key in child_key:
        if key in curr_action:
            child_key_name = key
    new_action = {child_key_name: {"has_name": span}, "action_type": action_type}
    previous_action["action_sequence"].insert(curr_index + 1, new_action)

    return previous_action


def fix_composite_in_dict(action_text, action_dict):
    """Find if action_dict has a schematic / reference_object that has two entitites
    in it, if so, split them and create two different action dicts for each."""

    action_description = [x.split() for x in action_text]
    name_1, name_2 = None, None
    sent_index = 0
    if "action_sequence" in action_dict:
        for i, curr_action in enumerate(action_dict["action_sequence"]):
            action_type = curr_action["action_type"]
            if action_type not in child_key_map:
                continue
            child_key = child_key_map[action_type]
            for key in child_key:
                if key in curr_action and "has_name" in curr_action[key]:
                    sent_index, indices = curr_action[key]["has_name"]
                    curr_sentence = action_text[sent_index]
                    curr_name = " ".join(curr_sentence.split()[indices[0] : indices[1] + 1])
                    if "__&&__" in curr_name:
                        # extract the two names
                        name_1, name_2 = curr_name.split("__&&__")
                        # fix the generated sentence
                        joining_word = random.choice([" and ", " and then "])
                        action_text[sent_index] = joining_word.join(curr_sentence.split("__&&__"))

                        # update the sentence split
                        action_description = [x.split() for x in action_text]

                        # fix this name's span
                        span = find_span(action_description, name_1)
                        curr_action[key]["has_name"] = span

                        # now add another action with the second name
                        action_dict = add_new_similar_action(
                            action_description, action_dict, name_2, curr_action, i
                        )
                        return action_dict

    return action_dict


def fix_spans(d, prev_sentence_len):
    """This function updates the spans in d and shifts them by a value equal
    to prev_sentence_len"""
    for key, val in d.items():
        if type(val) == list:
            if type(val[0]) == list:
                val = val[0]
            if type(val[0]) is int:
                sent_index, span = val
                index_1, index_2 = span
                d[key] = [sent_index, [index_1 + prev_sentence_len, index_2 + prev_sentence_len]]
            elif type(val[0]) is dict:
                for v in val:
                    if type(v) is dict:
                        fix_spans(v, prev_sentence_len)
        elif type(val) == dict:
            fix_spans(val, prev_sentence_len)

    return d


def combine_dicts(dict_1, dict_2, prev_sentence_len):
    """ This function appends the 'action_sequence' of dict_2 to dict_1
    after updating spans in dict_2 with length of sentence before it"""
    dict_2 = fix_spans(dict_2, prev_sentence_len)
    for action in dict_2["action_sequence"]:
        dict_1["action_sequence"].append(action)
    return dict_1


def create_composite_action(action_1, action_2):
    """This function takes in two actions, combines their texts together and
    combines their dicts"""

    text_1, text_2 = action_1.generate_description(), action_2.generate_description()
    dict_1, dict_2 = action_1.to_dict(), action_2.to_dict()

    # in case there are compostite schematics in either actions, expand action_sequence
    # to accomodate them
    dict_1 = fix_composite_in_dict(text_1, dict_1)
    dict_2 = fix_composite_in_dict(text_2, dict_2)

    # combine the sentences together
    prev_text = text_1[0] + random.choice([" and ", " and then "])
    composite_text = prev_text + text_2[0]
    prev_action_length = len(prev_text.split())

    # combine the dicts together
    composite_dict = combine_dicts(dict_1, dict_2, prev_action_length)
    return [composite_text], composite_dict


def generate_actions(n, action_type=None, template_attributes={}, composite=False):
    """ Generate action tree and language based on action type """

    texts = []
    dicts = []
    commands_generated = set()
    count = 0

    while count < n:
        # pick an action name
        action_name = (
            random.choice(action_type)
            if type(action_type) is list
            else action_type_map[action_type]
        )
        composite_flag = None

        # if None, no preference mentioned, pick True at random
        if composite is None:
            composite_flag = True if random.random() < 0.3 else False
        else:
            # assign preference (True or False)
            composite_flag = composite

        # if composite flag is True, generate composite action
        if composite_flag and action_name in human_give_command_actions:
            template_attributes["dialogue_len"] = 1
            # check how to optimize this call
            action_1 = Action.generate(action_type=action_name, template_attr=template_attributes)

            # pick another from human_give_command_actions
            possible_choices = human_give_command_actions
            next_action = random.choice(possible_choices)
            action_2 = Action.generate(action_type=next_action, template_attr=template_attributes)

            action_text, action_dict = create_composite_action(action_1, action_2)
        elif composite_flag == False:
            template_attributes["no_inbuilt_composites"] = True
            action_1 = Action.generate(action_type=action_name, template_attr=template_attributes)
            action_text = action_1.generate_description()
            action_dict = action_1.to_dict()
            action_dict = fix_composite_in_dict(action_text, action_dict)
        else:
            action_1 = Action.generate(action_type=action_name, template_attr=template_attributes)
            action_text = action_1.generate_description()
            action_dict = action_1.to_dict()
            action_dict = fix_composite_in_dict(action_text, action_dict)
        # else generate composite action at random
        full_command = " ".join(action_text)
        if full_command not in commands_generated:
            commands_generated.add(full_command)
            texts.append(action_text)
            dicts.append(action_dict)
            count += 1
    return texts, dicts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--chats_file", "-chats", type=str, default="noop_dataset.txt")
    parser.add_argument("--action_type", default=Action.CHOICES)
    parser.add_argument(
        "--composite_action",
        action="store_true",
        help="this flag tells the script to create composite actions only",
    )
    parser.add_argument(
        "--no_composite_actions",
        action="store_true",
        help="this flag tells the script to create no composite actions",
    )
    parser.add_argument(
        "--length_range", nargs="+", default=None, help="Low and High for range of length"
    )
    parser.add_argument(
        "--width_range", nargs="+", default=None, help="Low and High for range of width"
    )
    parser.add_argument(
        "--depth_range", nargs="+", default=None, help="Low and High for range of depth"
    )
    parser.add_argument(
        "--height_range", nargs="+", default=None, help="Low and High for range of height"
    )
    parser.add_argument(
        "--size_range", nargs="+", default=None, help="Low and High for range of size"
    )
    parser.add_argument(
        "--thickness_range", nargs="+", default=None, help="Low and High for range of thickness"
    )
    parser.add_argument(
        "--radius_range", nargs="+", default=None, help="Low and High for range of radius"
    )
    parser.add_argument(
        "--base_range", nargs="+", default=None, help="Low and High for range of base"
    )
    parser.add_argument(
        "--slope_range", nargs="+", default=None, help="Low and High for range of slope"
    )
    parser.add_argument(
        "--distance_range", nargs="+", default=None, help="Low and High for range of distance"
    )
    parser.add_argument(
        "--step_range", nargs="+", default=None, help="Low and High for range of steps"
    )
    parser.add_argument(
        "--coordinates_range",
        nargs="+",
        default=None,
        help="Low and High for range of coordinates",
    )
    parser.add_argument(
        "--count_range",
        nargs="+",
        default=None,
        help="Low and High for range of count / repetitions",
    )
    parser.add_argument(
        "--non_shape_names",
        type=str,
        default=None,
        help="The file containing names of supported schematics (non standard shapes)",
    )
    parser.add_argument(
        "--mob_file", type=str, default=None, help="The file containing names of supported mobs"
    )
    parser.add_argument(
        "--block_types_file",
        type=str,
        default=None,
        help="The file containing supported block objects",
    )

    args = parser.parse_args()

    # load file containing negative examples of chats
    try:
        f = open(args.chats_file)
        chats = [line.strip() for line in f]
        f.close()
        Noop.CHATS += [
            x
            for x in chats
            if x
            not in [
                "good job",
                "cool",
                "that is really cool",
                "that is awesome",
                "awesome",
                "that is amazing",
                "that looks good",
                "you did well",
                "great",
                "good",
                "nice",
                "that is wrong",
                "that was wrong",
                "that was completely wrong",
                "not that",
                "that looks horrible",
                "that is not what i asked",
                "that is not what i told you to do",
                "that is not what i asked for",
                "not what i told you to do",
                "you failed",
                "failure",
                "fail",
                "not what i asked for",
                "where are you",
                "tell me where you are",
                "i do n't see you",
                "i ca n't find you",
                "are you still around",
                "what are you doing",
                "now what are you building",
                "tell me what are you doing",
                "what is your task",
                "tell me your task",
                "what are you up to",
                "stop",
                "wait",
                "where are you going",
                "what is this",
                "come here",
                "mine",
                "what is that thing",
                "come back",
                "go back",
                "what is that",
                "keep going",
                "tower",
                "follow me",
                "do n't do that",
                "do n't move",
                "hold on",
                "this is pretty",
                "continue",
                "can you follow me",
                "move",
                "this is nice",
                "this is sharp",
                "this is very big",
                "keep digging",
                "circle",
                "that is sharp",
                "it looks nice",
            ]
        ]
    except:
        print("chats file not found")

    random.seed(args.seed)
    template_attributes = {}
    arg_dict = args.__dict__
    for key in arg_dict:
        if arg_dict[key]:
            # Assign numeric ranges
            if key in arg_numeric_range_map:
                low, high = [int(x) for x in arg_dict[key]]
                template_attributes[arg_numeric_range_map[key]] = range(low, high)
            # Assign names of schematic and reference objects
            elif key in arg_name_file_map:
                with open(arg_dict[key]) as f:
                    template_attributes[arg_name_file_map[key]] = [
                        line.strip() for line in f.readlines()
                    ]
    composite_flag = None
    # assign composite_flag if user explcitly mentioned to create / avoid them
    # else pick True at random inside generate_actions()
    if args.composite_action:
        composite_flag = True
    if args.no_composite_actions:
        composite_flag = False
    for text, d in zip(
        *generate_actions(
            args.n,
            args.action_type,
            template_attributes=template_attributes,
            composite=composite_flag,
        )
    ):
        for sentence in text:
            print(sentence)
        print(json.dumps(d))
        print()
