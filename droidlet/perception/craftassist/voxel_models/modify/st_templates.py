"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import random
import sys
import argparse

from .. import minecraft_specs
from ..shape_helpers import SHAPE_NAMES

ID_DELIM = "^"
BLOCK_NAMES = [v for k, v in minecraft_specs.get_block_data()["bid_to_name"].items() if k[0] < 256]
COLOR_NAMES = [
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "green",
    "gray",
    "lime",
    "maroon",
    "navy",
    "olive",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
    "orange",
    "brown",
    "sienna",
    "pink",
    "light yellow",
    "dark yellow",
    "dark yellow",
    "gold",
    "gold",
]
# COLOR_DATA = minecraft_specs.get_colour_data()


def build_lf(ref_obj_dict, modify_dict):
    action_dict = {"action_type": "MODIFY", "modify_dict": modify_dict}
    if ref_obj_dict is not None:
        action_dict["reference_object"] = ref_obj_dict
    y = {"dialogue_type": "HUMAN_GIVE_COMMAND", "action_sequence": [action_dict]}
    return y


def replace_with_span(d, split_text):
    if type(d) is dict:
        for k, v in d.items():
            if type(v) is str:
                vsplit = v.split()
                identifier = vsplit[0].split(ID_DELIM)
                if len(identifier) > 1:
                    identifier = identifier[1]
                    has_id = [i for i, word in enumerate(split_text) if identifier in word]
                    span = [0, [min(has_id), max(has_id)]]
                    d[k] = {"span": span}
                    for i in has_id:
                        split_text[i] = split_text[i].strip(ID_DELIM + identifier)
            else:
                replace_with_span(v, split_text)
    else:
        return


def get_target_object():
    shape_name = random.choice(SHAPE_NAMES).lower()
    shape_name_split = shape_name.split("_")
    rid = str(random.random())
    object_text = " ".join([v + ID_DELIM + rid for v in shape_name_split])
    ref_obj = {"filters": {"has_name": object_text}}
    ref_obj_text = "the " + object_text
    return ref_obj, ref_obj_text


def get_target_location():
    loc_dict = {"location": {"location_type": "SPEAKER_LOOK"}}
    loc_text = "there"
    return loc_dict, loc_text


def get_block():
    rid = str(random.random())
    if random.random() < 0.5:
        csplit = random.choice(COLOR_NAMES).split()
        colour = " ".join([w + ID_DELIM + rid for w in csplit])
        block_dict = {"has_colour": colour}
        block_text = colour + " blocks"
    else:
        bsplit = random.choice(BLOCK_NAMES).split()
        blockname = " ".join([w + ID_DELIM + rid for w in bsplit])
        block_dict = {"has_name": blockname}
        block_text = blockname

    return block_dict, block_text


# THICKEN/SCALE/RIGIDMOTION/REPLACE/FILL


class ModifyTemplates:
    def __init__(self):
        pass

    def generate(self):
        pass


class ThickenTemplates(ModifyTemplates):
    def __init__(self, opts):
        pass

    def generate(self):
        ref_obj, ref_obj_text = get_target_object()
        modify_text = "make " + ref_obj_text
        if random.random() > 0.5:
            modify_text += " thicker"
            modify_dict = {"modify_type": "THICKER"}
        else:
            modify_text += " thinner"
            modify_dict = {"modify_type": "THINNER"}
        return modify_text, modify_dict, ref_obj_text, ref_obj


class ScaleTemplates(ModifyTemplates):
    def __init__(self, opts):
        self.not_makephrase = 0.5

    def generate(self):
        ref_obj, ref_obj_text = get_target_object()
        s = random.choice(
            ["WIDER", "NARROWER", "TALLER", "SHORTER", "SKINNIER", "FATTER", "BIGGER", "SMALLER"]
        )
        modify_dict = {"modify_type": "SCALE", "categorical_scale_factor": s}
        modify_text = "make " + ref_obj_text + " " + s.lower()
        if random.random() < self.not_makephrase:
            if s == "WIDER":
                modify_text = "widen " + ref_obj_text
            elif s == "NARROWER":
                modify_text = "narrow " + ref_obj_text
            elif s == "SHORTER":
                modify_text = "shorten " + ref_obj_text
            elif s == "FATTER":
                modify_text = "fatten " + ref_obj_text
            elif s == "BIGGER":
                modify_text = (
                    random.choice(["upscale ", "grow ", "increase the size of "]) + ref_obj_text
                )
            elif s == "SMALLER":
                modify_text = "shrink " + ref_obj_text

        return modify_text, modify_dict, ref_obj_text, ref_obj


class RigidmotionTemplates(ModifyTemplates):
    def __init__(self, opts):
        self.opts = opts

    def generate(self):
        ref_obj, ref_obj_text = get_target_object()
        modify_dict = {"modify_type": "RIGIDMOTION"}
        if random.random() < self.opts.translate_prob:
            loc_dict, loc_text = get_target_location()
            modify_dict["location"] = loc_dict
            modify_text = random.choice(["move ", "put "]) + ref_obj_text + " " + loc_text
        else:
            if random.random() < self.opts.flip_prob:
                modify_dict["mirror"] = True
                modify_text = random.choice(["flip ", "mirror "]) + ref_obj_text
            else:
                d = random.choice(["LEFT", "RIGHT", "AROUND"])
                modify_dict["categorical_angle"] = d
                modify_text = random.choice(["rotate ", "turn "]) + ref_obj_text + " " + d.lower()

        return modify_text, modify_dict, ref_obj_text, ref_obj


class ReplaceTemplates(ModifyTemplates):
    def __init__(self, opts):
        self.opts = opts

    def generate(self):
        ref_obj, ref_obj_text = get_target_object()
        modify_dict = {"modify_type": "REPLACE"}
        new_block_dict, new_block_text = get_block()
        t = random.choice(["make |", "replace with", "swap with", "change to"]).split()
        if random.random() < self.opts.old_blocktype:
            # TODO "all"
            old_block_dict, old_block_text = get_block()
            modify_text = (
                t[0] + " the " + old_block_text + " " + t[1].strip("|") + " " + new_block_text
            )
            modify_dict["old_block"] = old_block_dict
            if random.random() > 0.5:
                modify_text += " in the " + ref_obj_text
            else:
                ref_obj = None
        else:
            # TODO geom *and* blocktype, every n
            d = random.choice(["LEFT", "RIGHT", "TOP", "BOTTOM", "FRONT", "BACK"])
            fraction = random.choice(["QUARTER", "HALF", ""])
            if fraction == "":
                modify_dict["replace_geometry"] = {"relative_direction": d.lower()}
                modify_text = (
                    t[0]
                    + " the "
                    + d.lower()
                    + " of "
                    + ref_obj_text
                    + " "
                    + t[1].strip("|")
                    + " "
                    + new_block_text
                )
            else:
                modify_dict["replace_geometry"] = {"relative_direction": d.lower()}
                modify_text = (
                    t[0]
                    + " the "
                    + d.lower()
                    + " "
                    + fraction.lower()
                    + " of "
                    + ref_obj_text
                    + " "
                    + t[1].strip("|")
                    + " "
                    + new_block_text
                )
        modify_dict["new_block"] = new_block_dict
        return modify_text, modify_dict, ref_obj_text, ref_obj


class FillTemplates(ModifyTemplates):
    def __init__(self, opts):
        pass

    def generate(self):
        ref_obj, ref_obj_text = get_target_object()
        if random.random() > 0.5:
            modify_text = "fill up the " + ref_obj_text
            modify_dict = {"modify_type": "FILL"}
            if random.random() > 0.5:
                new_block_dict, new_block_text = get_block()
                modify_dict["new_block"] = new_block_dict
                modify_text += " with " + new_block_text
        else:
            modify_text = "hollow out the " + ref_obj_text
            modify_dict = {"modify_type": "HOLLOW"}
        return modify_text, modify_dict, ref_obj_text, ref_obj


class TemplateHolder:
    def __init__(self, opts):
        # TODO
        #        self.gen_weights = opts.gen_weights
        self.templates = {
            "thicken": ThickenTemplates(opts),
            "scale": ScaleTemplates(opts),
            "rigidmotion": RigidmotionTemplates(opts),
            "replace": ReplaceTemplates(opts),
            "fill": FillTemplates(opts),
        }

    def generate(self):
        modify_text, modify_dict, ref_obj_text, ref_obj_dict = random.choice(
            list(self.templates.values())
        ).generate()
        split_modify_text = modify_text.split()
        replace_with_span(modify_dict, split_modify_text)
        if ref_obj_dict:
            replace_with_span(ref_obj_dict, split_modify_text)
        modify_text = " ".join(split_modify_text)
        return modify_text, build_lf(ref_obj_dict, modify_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        default="/checkpoint/rebeccaqian/datasets/modify_templates/",
        type=str,
        help="where to write modify data",
    )
    parser.add_argument(
        "--translate_prob", default=0.25, type=int, help="where to write modify data"
    )
    parser.add_argument("--flip_prob", default=0.1, type=int, help="where to write modify data")
    parser.add_argument(
        "--old_blocktype", default=0.25, type=str, help="where to write modify data"
    )
    parser.add_argument("-N", default=100, type=int, help="number of samples to generate")
    opts = parser.parse_args()
    T = TemplateHolder(opts)
    data = []
    for i in range(opts.N):
        data.append(T.generate())
    f = open(opts.target_dir + "templated_modify.txt", "w")
    for d in data:
        cmd, action_dict = d
        f.write("{}|{}\n".format(cmd, action_dict))


if __name__ == "__main__":
    main()
