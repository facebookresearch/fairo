"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import random
from generate_dialogue import generate_actions
import sys
import os
import uuid
import json

TTAD_GEN_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(TTAD_GEN_DIR, "../../")
sys.path.append(CRAFTASSIST_DIR)
from generate_data import *
import re
from dialogue_objects.interpreter_helper import coref_resolve, interpret_shape_schematic
from word_maps import SPECIAL_SHAPE_FNS, SPECIAL_SHAPES_CANONICALIZE, SPAWN_OBJECTS
from size_words import size_str_to_int
import shapes
import block_data
import snowballstemmer

stemmer = snowballstemmer.stemmer("english")

# from word2number.w2n import word_to_num

CHOICES = [Move, Build, Destroy, Dig, Copy, Fill, Spawn, Dance]


#############################################################
# modified from size_words...
#############################################################

RANGES = {
    "tiny": (2, 3),
    "small": (2, 3),
    "medium": (2, 4),
    "large": (4, 5),
    "huge": (5, 6),
    "gigantic": (5, 6),
}


class Dummy:
    pass


def inspect_ads(template_attributes, condition):
    while True:
        a = get_good_ad(template_attributes)
        if eval(condition):
            print(a[0])
            print(json.dumps(a[1], sort_keys=True, indent=4))
            break
    return a


def process_spans(d, words):
    for k, v in d.items():
        if type(v) == dict:
            process_spans(v, words)
            continue
        try:
            if k != "has_attribute":
                sentence, (L, R) = v
                if sentence != 0:
                    raise NotImplementedError("Must update process_spans for multi-string inputs")
            assert 0 <= L <= R <= (len(words) - 1)
        except ValueError:
            continue
        except TypeError:
            continue

        d[k] = " ".join(words[L : (R + 1)])


class EmptyWorkspaceMemory:
    def get_recent_entities(self, x=None):
        return []


def surgery_by_value(ad, old_value, new_value):
    for k, v in ad.items():
        if type(v) is dict:
            surgery_by_value(v, old_value, new_value)
        else:
            if v == old_value:
                ad[k] = new_value


def get_fields_by_key(ad_and_parent, key):
    parent = ad_and_parent[0]
    ad = ad_and_parent[1]

    if type(ad) is not dict:
        return []
    else:
        values = []
        for k, v in ad.items():
            u = get_fields_by_key((k, v), key)
            values.extend(u)
            if k == key:
                values.append((parent, v))
        return values


# TODO make sure if object has attributes and a location
# e.g. speakerlook there isn't a conflict ?


def get_good_ad(template_attributes, flat=False):
    ok = False
    while not ok:
        ad_and_text = generate_actions(1, CHOICES, template_attributes)
        ok = True
        if len(ad_and_text[0][0]) > 1:
            # filter dialogues
            ok = False
            continue
        text = ad_and_text[0][0][0]
        ad = ad_and_text[1][0]["action_sequence"][0]
        atpd = template_attributes.get("distribution")
        if atpd is not None:
            if np.random.rand() > atpd[ad["action_type"]]:
                ok = False
                continue
        fill_spans_and_coref_resolve(ad, text)
        ref_objs = get_fields_by_key((None, ad), "reference_object")
        if len(ref_objs) > 0:
            for r in ref_objs:
                p = r[0]
                o = r[1]
                # filter things like 'go above the 5 cubes'
                if p == "location" and o.get("repeat") is not None:
                    ok = False
                # filter things like 'destroy the 5 cubes' (unfortunately killing 'all' rn too, FIXME)
                if ad["action_type"] == "DESTROY" and o.get("repeat") is not None:
                    ok = False
                # filter things like 'copy the 5 cubes' (unfortunately killing 'all' rn too, FIXME)
                if ad["action_type"] == "BUILD" and o.get("repeat") is not None:
                    ok = False
        # filter "between" FIXME!!!! probably should do two-object betweens
        #        betweens = get_fields_by_key((None, ad), "reference_object_1")
        #        if len(betweens)>0:
        #            ok = False
        # but one object one would lead to bias bc of duplicate
        betweens = get_fields_by_key((None, ad), "relative_direction")
        if len(betweens) > 0 and betweens[0][1] == "BETWEEN":
            ok = False
        # filter exact coordinates
        c = get_fields_by_key((None, ad), "coordinates")
        if len(c) > 0:
            ok = False
        # filter stop conditions of 'ADJACENT_TO_BLOCK_TYPE'
        c = get_fields_by_key((None, ad), "condition_type")
        if len(c) > 0 and c[0][1] == "ADJACENT_TO_BLOCK_TYPE":
            ok = False

        # FOR NOW, FILTERING RELDIRS UP, DOWN, INSIDE, OUTSIDE, AWAY!!! FIXME!!
        c = get_fields_by_key((None, ad), "relative_direction")
        if len(c) > 0 and c[0][1] in ["UP", "DOWN", "INSIDE", "OUTSIDE", "AWAY"]:
            ok = False

        r = get_fields_by_key((None, ad), "repeat_key")
        # filter finite repeats of move actions ("move three steps twice")
        if ad["action_type"] == "MOVE" and len(r) > 0 and r[0][1] == "FOR":
            ok = False

        # filter large objects
        #        s = get_fields_by_key((None, ad), "has_size")
        #        for i in s:
        #            for size in ["gigantic", "huge", "colossal", "large", "big"]:
        #                if size in i[1]:
        #                    new_size = random.choice(["medium", "small"])
        #                    surgery_by_value(ad, size, new_size)
        #                    text = text.replace(size, new_size)

        r = get_fields_by_key((None, ad), "schematic")
        if flat:
            if len(r) > 0:
                for s in r:
                    name = s[1].get("has_name")
                    if name is not None:
                        # FIXME this is not grammatical...
                        new_name = random.choice(template_attributes["non_shape_names"])
                        surgery_by_value(ad, name, new_name)
                        text = text.replace(name, new_name)

        r = get_fields_by_key((None, ad), "has_block_type")
        if len(r) > 0:
            allowed_blocks = template_attributes.get("allowed_blocktypes")
            if allowed_blocks:
                for (_, btype) in r:
                    if btype not in allowed_blocks:
                        new_btype = random.choice(allowed_blocks)
                        text = text.replace(btype, new_btype)
                        surgery_by_value(ad, btype, new_btype)

        # filter empty builds and empty moves:
    #        if ad["action_type"] == "MOVE": TODO

    return text, ad, ref_objs


def fill_spans_and_coref_resolve(ad, text):
    W = EmptyWorkspaceMemory()
    process_spans(ad, re.split(r" +", text))
    coref_resolve(W, ad)
    return ad


def regularize_ref_obj_dict(ref_obj_dict, flat=False):
    # TODO make me optional
    if ref_obj_dict.get("has_colour") is None:
        ref_obj_dict["has_colour"] = random.choice(list(block_data.COLOR_BID_MAP.keys()))
    if flat:
        ref_obj_dict["has_orientation"] = "xz"
        if ref_obj_dict.get("has_size") is None:
            ref_obj_dict["has_size"] = "medium"


# takes the piece of the action dict and makes it into a game object
# does not put locs, these need to be fixed after
def specify_object(ref_obj_dict, sl=32, flat=False):
    shape_keys = [
        "has_size",
        "has_thickness",
        "has_radius",
        "has_depth",
        "has_width",
        "has_height",
        "has_length",
        "has_slope",
        "has_orientation",
        "has_distance",
        "has_base",
        "has_colour",
    ]

    ##############################################
    # hack to deal with size ranges in size_words
    ##############################################
    fake_interpreter = Dummy()
    fake_interpreter.agent = Dummy()

    def ssti(s):
        return size_str_to_int(s, ranges=RANGES)

    fake_interpreter.agent.size_str_to_int = ssti

    name = ref_obj_dict.get("has_name")
    if name is not None:
        stemmed_name = stemmer.stemWord(name)
        shapename = SPECIAL_SHAPES_CANONICALIZE.get(name) or SPECIAL_SHAPES_CANONICALIZE.get(
            stemmed_name
        )

        if SPAWN_OBJECTS.get(name):
            return name
        elif SPAWN_OBJECTS.get(stemmed_name):
            return stemmed_name

        if shapename:
            regularize_ref_obj_dict(ref_obj_dict, flat=flat)
            blocks, _ = interpret_shape_schematic(
                fake_interpreter, None, ref_obj_dict, shapename=shapename
            )
            return blocks
        else:
            # somethings wrong, abort
            return None
    else:
        if ref_obj_dict.get("has_shape"):
            regularize_ref_obj_dict(ref_obj_dict, flat=flat)
            blocks, _ = interpret_shape_schematic(fake_interpreter, None, ref_obj_dict)
            return blocks
        elif any(k in shape_keys for k in ref_obj_dict.keys()):
            regularize_ref_obj_dict(ref_obj_dict, flat=flat)
            if flat:
                shapename = random.choice(["TRIANGLE", "CIRCLE", "DISK", "RECTANGLE"])
            else:
                shapename = random.choice(list(SPECIAL_SHAPE_FNS.keys()))
            blocks, _ = interpret_shape_schematic(
                fake_interpreter, None, ref_obj_dict, shapename=shapename
            )
            return blocks
        else:
            # somethings wrong, abort
            return None


def choose_loc(obj_rad, reldir, viewer_pos, view_vector, ref_obj_loc, sl=32, num_tries=100):
    reldir_perp = np.array((-view_vector[1], view_vector[0]))
    tform = np.stack((reldir_perp, view_vector), axis=1)
    obj_coords = tform.transpose() @ (ref_obj_loc - viewer_pos)
    for i in range(num_tries):
        loc = np.random.randint(-sl // 2, sl // 2, size=(2))
        loc_coords = tform.transpose() @ (loc - viewer_pos)
        if reldir == "LEFT":
            if loc_coords[0] > obj_coords[0] + obj_rad:
                return loc
        if reldir == "RIGHT":
            if loc_coords[0] < obj_coords[0] - obj_rad:
                return loc
        if reldir == "BACK":
            if loc_coords[1] < obj_coords[1] - obj_rad:
                return loc
        if reldir == "FRONT":
            if loc_coords[1] > obj_coords[1] + obj_rad:
                return loc
        if reldir == "NEAR":
            # 4 is arbitrary
            if max(abs(np.subtract(loc, ref_obj_loc))) - obj_rad < 4:
                return loc
    return None


def maybe_speaker_look(ref_obj_dict, view_pos, sl):
    location_type = ref_obj_dict.get("location", {}).get("location_type")
    if location_type is not None and location_type == "SPEAKER_LOOK":
        loc = np.add(view_pos, np.random.randint(-sl // 8, sl // 8, size=(2)))
    else:
        loc = np.random.randint(-sl // 2, sl // 2, size=(2))
    return loc


# step 1, arrange all reference objects, holes to fill and player
# step 2, build decoys.  for each ref object with some attributes (including a reference direction to another ref object)
#           build/place a decoy with some attributes the same
# step 3, build random objects/mobs/pits


def build_ref_obj_scene(ad, ref_objs, sl=32, flat=False):
    # first place the fake observer, we put them on a ring 75% away from the center of the cube, at ground level
    c = np.random.randn(2)
    c /= np.linalg.norm(c)
    radius = 0.75 * sl / 2
    p = radius * c
    # the fake observer will look at the ground somewhere along a cone starting from the observer pointing towards
    # the center of the square

    #    n = np.random.randn(2)
    #    n /= np.linalg.norm(n)
    #    n *= 0.05
    #    view_vector = -c + n
    view_vector = -c
    view_vector = view_vector / np.linalg.norm(
        view_vector
    )  # unit vector in the direction of my look
    d = radius + radius * np.random.uniform()
    view_pos = p + d * view_vector  # which grid location/cube (at ground height) am I looking at?
    scene = {"mobs": [], "block_obj": [], "holes": []}
    scene["fake_human"] = {"position": p, "view_vector": view_vector, "view_pos": view_pos}
    scene["action_dict"] = ad
    scene["ref_obj_dicts"] = {}

    at = ad["action_type"]

    if len(ref_objs) == 1 and at != "FILL" and at != "SPAWN":
        # FIXME if loc is given in ad
        loc = maybe_speaker_look(ref_objs[0][1], view_pos, sl)
        obj = specify_object(ref_objs[0][1], sl=sl, flat=flat)
        robj_id = uuid.uuid4().hex
        scene["ref_obj_dicts"][robj_id] = ref_objs[0][1]
        if type(obj) is str:
            scene["mobs"].append({"mobname": obj, "loc": loc, "dict_id": robj_id})
            return scene
        else:
            scene["block_obj"].append({"blocks": obj, "loc": loc, "dict_id": robj_id})
            return scene

    if len(ref_objs) == 1 and at == "FILL":
        obj_loc = maybe_speaker_look(ref_objs[0][1], view_pos, sl)
        obj = specify_object(ref_objs[0][1], sl=sl, flat=flat)
        reldir = ad["location"]["relative_direction"]
        robj_id = uuid.uuid4().hex
        scene["ref_obj_dicts"][robj_id] = ref_objs[0][1]
        if type(obj) is str:
            scene["mobs"].append({"mobname": obj, "loc": obj_loc, "dict_id": robj_id})
            hole_loc = choose_loc(1, reldir, view_pos, view_vector, obj_loc, sl=sl)
            scene["holes"].append({"loc": hole_loc})
        elif obj is not None and len(obj) > 0:
            scene["block_obj"].append({"blocks": obj, "loc": obj_loc, "dict_id": robj_id})
            bounds = shapes.get_bounds(obj)
            rad = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            hole_loc = choose_loc(rad, reldir, view_pos, view_vector, obj_loc, sl=sl)
            scene["holes"].append({"loc": hole_loc})
        # TODO fix relative location; put this in choose_loc, input the ref_obj
        elif ref_objs[0][1].get("location")["location_type"] == "SPEAKER_LOOK":
            scene["holes"].append({"loc": view_pos.astype("int64")})
        elif ref_objs[0][1].get("location")["location_type"] == "SPEAKER_POS":
            scene["holes"].append({"loc": p.astype("int64")})

        return scene

    if len(ref_objs) == 2:
        if at == "DESTROY":
            reldir = ad["reference_object"]["location"]["relative_direction"]
            if ref_objs[0][0] == "action":
                d = ref_objs[0][1]
                r = ref_objs[1][1]
            else:
                d = ref_objs[1][1]
                r = ref_objs[0][1]
            d_id = uuid.uuid4().hex
            r_id = uuid.uuid4().hex
            scene["ref_obj_dicts"][d_id] = d
            scene["ref_obj_dicts"][r_id] = r
            to_destroy_obj = specify_object(d, sl=sl, flat=flat)
            rel_obj = specify_object(r, sl=sl, flat=flat)
            rel_obj_loc = maybe_speaker_look(r, view_pos, sl)
            if type(rel_obj) is str:
                rad = 1
                scene["mobs"].append({"mobname": rel_obj, "loc": rel_obj_loc, "dict_id": r_id})
            else:
                scene["block_obj"].append({"blocks": rel_obj, "loc": rel_obj_loc, "dict_id": r_id})
                bounds = shapes.get_bounds(rel_obj)
                rad = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            to_destroy_loc = choose_loc(rad, reldir, view_pos, view_vector, rel_obj_loc, sl=sl)
            scene["block_obj"].append(
                {"blocks": to_destroy_obj, "loc": to_destroy_loc, "dict_id": d_id}
            )
            return scene

        # this is a copy, one ref object is to be copied, and the other gives
        # the *target* location, so locations are independent
        elif at == "BUILD":
            if ref_objs[0][0] == "location":
                c = ref_objs[1][1]
                r = ref_objs[0][1]
            else:
                c = ref_objs[0][1]
                r = ref_objs[1][1]

            c_id = uuid.uuid4().hex
            r_id = uuid.uuid4().hex
            scene["ref_obj_dicts"][c_id] = c
            scene["ref_obj_dicts"][r_id] = r
            to_copy_obj = specify_object(c, sl=sl, flat=flat)
            to_copy_obj_loc = maybe_speaker_look(c, view_pos, sl)
            rel_obj = specify_object(r, sl=sl, flat=flat)
            rel_obj_loc = np.random.randint(-sl // 2, sl // 2, size=(2))
            scene["block_obj"].append(
                {"blocks": to_copy_obj, "loc": to_copy_obj_loc, "dict_id": c_id}
            )
            if type(rel_obj) is str:
                scene["mobs"].append({"mobname": rel_obj, "loc": rel_obj_loc, "dict_id": r_id})
            else:
                scene["block_obj"].append({"blocks": rel_obj, "loc": rel_obj_loc, "dict_id": r_id})
            return scene
    return scene


def add_distractors(
    scene, template_attributes, sl=32, num_objs=3, num_holes=2, num_mobs=2, flat=False
):
    while (
        len(scene["block_obj"]) < num_objs
        or len(scene["holes"]) < num_holes
        or len(scene["mobs"]) < num_mobs
    ):
        text, ad, ref_objs = get_good_ad(template_attributes, flat=flat)
        distractor_scene = build_ref_obj_scene(ad, ref_objs, sl=sl, flat=flat)
        if distractor_scene is not None:
            # not careful about overlaps... should we be?
            for bobj in distractor_scene["block_obj"]:
                if len(scene["block_obj"]) < num_objs:
                    scene["block_obj"].append(bobj)
                    k = bobj["dict_id"]
                    v = distractor_scene["ref_obj_dicts"][k]
                    scene["ref_obj_dicts"][bobj["dict_id"]] = v
            for hole in distractor_scene["holes"]:
                if len(scene["holes"]) < num_holes:
                    scene["holes"].append(hole)
            for mob in distractor_scene["mobs"]:
                if len(scene["mobs"]) < num_mobs:
                    scene["mobs"].append(mob)
                    k = mob["dict_id"]
                    v = distractor_scene["ref_obj_dicts"][k]
                    scene["ref_obj_dicts"][mob["dict_id"]] = v


def get_slice(blocks, axis=0, coord=0):
    return [b for b in blocks if b[0][axis] == coord]


def build_scene(template_attributes, sl=32, flat=False):
    text, ad, ref_objs = get_good_ad(template_attributes, flat=flat)
    S = build_ref_obj_scene(ad, ref_objs, sl=sl, flat=flat)
    if S is not None:
        S["non_distractors"] = []
        for o in S["ref_obj_dicts"]:
            S["non_distractors"].append(o)
        S["text"] = text
        add_distractors(S, template_attributes, sl=sl, flat=flat)
        S["id_to_obj"] = {}
        for m in S["mobs"]:
            S["id_to_obj"][m["dict_id"]] = m
        for m in S["block_obj"]:
            S["id_to_obj"][m["dict_id"]] = m
        if flat:
            for b in S["block_obj"]:
                if b["blocks"] is not None:
                    b["blocks"] = get_slice(b["blocks"], axis=1, coord=0)

    return S


if __name__ == "__main__":

    template_attributes = {"count": range(1, 5)}
    template_attributes["step"] = range(1, 10)
    template_attributes["non_shape_names"] = list(SPECIAL_SHAPES_CANONICALIZE.keys())
    template_attributes["mob_names"] = ["pig", "sheep", "cow", "chicken", "rabbit"]

    for i in range(1000):
        print(i)
        S = build_scene(template_attributes)


# import json
# x = {'a':{'at':'d', 'R':{'hs':'h', 'ha':'h', 'l':{'rd':'right', 'lt':'RO','R':{'hn':'donk'}}}}}
# print(json.dumps(x, sort_keys=True, indent=4))
# u = get_fields_by_key((None, x) , 'R')
# print(len(u))
