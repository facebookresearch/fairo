"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from preprocess import word_tokenize
from ttad.generation_dialogues.generate_dialogue import action_type_map  # Dict[str, Class]
from ttad.generation_dialogues.templates.templates import template_map  # Dict[str, List[Template]]

from typing import Dict, List, Tuple, Sequence


def adtt(d: Dict) -> str:
    """Return a string that would produce the action dict `d`

    d is post-process_span (i.e. its span values are replaced with strings)
    d is pre-coref_resolve (i.e. its coref_resolve values are strings, not
        memory objects or keywords)

    """
    if d["dialogue_type"] != "HUMAN_GIVE_COMMAND":
        raise NotImplementedError("can't handle {}".format(d["dialogue_type"]))

    action_type = d["action"]["action_type"]  # e.g. "MOVE"
    action_type = action_type[0].upper() + action_type[1:].lower()  # e.g. "Move"

    for template in template_map[action_type]:
        dialogue, gen_d = generate_from_template(action_type, template)
        recurse_remove_keys(gen_d, ["has_attribute"])
        if len(dialogue) != 1:
            continue
        if dicts_match(d, gen_d):
            print(gen_d)
            text = replace_spans(dialogue[0], gen_d, d)
            print(dialogue[0])
            return replace_relative_direction(text, gen_d, d)

    raise ValueError("No matching template found for {}".format(d))


def replace_spans(text: str, gen_d: Dict, d: Dict) -> str:
    """Replace words in text with spans from d"""

    words = word_tokenize(text).split()

    # compile list of spans to replace via recursive search
    replaces = []
    to_consider = [(gen_d, d)]
    while len(to_consider) > 0:
        cur_gen_d, cur_d = to_consider.pop()
        for k in cur_gen_d.keys():
            if type(cur_d[k]) == dict:
                to_consider.append((cur_gen_d[k], cur_d[k]))
            elif type(cur_d[k]) == str and cur_d[k].upper() != cur_d[k]:
                replaces.append((cur_gen_d[k], cur_d[k]))

    # replace each span in words
    replaces.sort(key=lambda r: r[0][1][0], reverse=True)  # sort by L of span
    for (sentence_idx, (L, R)), s in replaces:
        assert sentence_idx == 0
        words = words[:L] + word_tokenize(s).split() + words[(R + 1) :]

    return " ".join(words)


def generate_from_template(action_type: str, template: List) -> Tuple[List[str], Dict]:
    cls = action_type_map[action_type.lower()]
    node = cls.generate(template)
    dialogue = node.generate_description()
    d = node.to_dict()
    return dialogue, d


def dicts_match(
    d: Dict,
    e: Dict,
    ignore_values_for_keys: Sequence[str] = ["relative_direction"],
    ignore_keys: Sequence[str] = ["has_attribute"],
) -> bool:
    if (set(d.keys()) - set(ignore_keys)) != (set(e.keys()) - set(ignore_keys)):
        return False

    for k, v in d.items():
        if type(v) == dict and not dicts_match(v, e[k]):
            return False

        # allow values of certain keys to differ (e.g. relative_direction)
        # allow spans (lowercase strings) to differ
        if (
            k not in ignore_keys
            and k not in ignore_values_for_keys
            and type(v) == str
            and v == v.upper()
            and v != e[k]
        ):
            return False

    return True


def recurse_remove_keys(d: Dict, keys: Sequence[str]):
    # remove keys from dict
    for x in keys:
        if x in d:
            del d[x]

    # recurse
    for k, v in d.items():
        if type(v) == dict:
            recurse_remove_keys(v, keys)


def replace_relative_direction(text: str, gen_d: Dict, d: Dict) -> str:
    try:
        rel_dir = d["action"]["location"]["relative_direction"]
        agent_pos = False
        try:
            if (
                d["action"]["location"]["reference_object"]["location"]["location_type"]
                == "AGENT_POS"
            ):
                agent_pos = True
        except:
            agent_pos = False

        # generate direction dict
        direction_dict = {}
        if not agent_pos:
            direction_dict["LEFT"] = ["to the left of", "towards the left of"]
            direction_dict["RIGHT"] = ["to the right of", "towards the right of"]
            direction_dict["UP"] = ["above", "on top of", "to the top of"]
            direction_dict["DOWN"] = ["below", "under"]
            direction_dict["FRONT"] = ["in front of"]
            direction_dict["BACK"] = ["behind"]
            direction_dict["AWAY"] = ["away from"]
            direction_dict["INSIDE"] = ["inside"]
            direction_dict["OUTSIDE"] = ["outside"]
            direction_dict["NEAR"] = ["next to", "close to", "near"]
            direction_dict["CLOCKWISE"] = ["clockwise"]
            direction_dict["ANTICLOCKWISE"] = ["anticlockwise"]
        else:
            direction_dict["LEFT"] = ["to the left", "to your left", "east", "left"]
            direction_dict["RIGHT"] = ["to the right", "to your right", "right", "west"]
            direction_dict["UP"] = ["up", "north"]
            direction_dict["DOWN"] = ["down", "south"]
            direction_dict["FRONT"] = ["front", "forward", "to the front"]
            direction_dict["BACK"] = ["back", "backwards", "to the back"]
            direction_dict["AWAY"] = ["away"]
            direction_dict["CLOCKWISE"] = ["clockwise"]
            direction_dict["ANTICLOCKWISE"] = ["anticlockwise"]

        # generate a list of the direction phrases and sort by longest to shortest
        direction_list: List[str] = []
        for k in direction_dict.keys():
            direction_list = direction_list + direction_dict[k]
        direction_list = sorted(direction_list, key=len, reverse=True)

        # look for direction phrase in the text to replace
        for dir_phrase in direction_list:
            if dir_phrase in text:
                text = text.replace(dir_phrase, direction_dict[rel_dir][0])
                break
        return text
    except:
        return text


if __name__ == "__main__":
    d = {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action": {
            "action_type": "BUILD",
            "schematic": {"has_name": "barn"},
            "location": {
                "location_type": "REFERENCE_OBJECT",
                "relative_direction": "LEFT",
                "reference_object": {"has_name": "boat house"},
            },
        },
    }
    t = adtt(d)
    print(t)
