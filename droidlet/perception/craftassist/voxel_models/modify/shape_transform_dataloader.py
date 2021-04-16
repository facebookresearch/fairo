"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import random
import torch
from torch.utils import data as tds
import pickle
import shape_transforms
import shape_helpers as sh
from build_utils import blocks_list_to_npy

from block_data import COLOR_BID_MAP

# Fetch block names from lowlevel/minecraft/craftassist_specs.get_block_data()
BLOCK_DATA = {}
# FIXME....
NEW_BLOCK_CHOICES = [idm for v in COLOR_BID_MAP.values() for idm in v]
BID_TO_COLOR = {idm: c for c in COLOR_BID_MAP.keys() for idm in COLOR_BID_MAP[c]}


##############################################
# WARNING: all npy arrays in this file are xyz
# not yzx


def rand():
    return torch.rand(1).item()


def thicker(schematic):
    data = {}
    data["tform_data"] = {"delta": int(np.floor(2 * rand()) + 1)}
    # FIXME prob
    if rand() > 0.5:
        thick_or_thin = "thicker"
        data["inverse"] = False
    else:
        thick_or_thin = "thinner"
        data["inverse"] = True
    text = "make it " + thick_or_thin
    tform = shape_transforms.thicker
    return tform, text, data


def scale(schematic):
    data = {}
    data["tform_data"] = {}
    d = random.choice(
        ["wider", "narrower", "taller", "shorter", "fatter", "skinnier", "bigger", "smaller"]
    )
    text = "make it " + d
    if d == "wider" or d == "narrower":
        # FIXME prob
        scale_factor = rand() + 1.0
        if rand() > 0.5:
            data["tform_data"]["lams"] = (1.0, 1.0, scale_factor)
        else:
            data["tform_data"]["lams"] = (scale_factor, 1.0, 1.0)
        if d == "wider":
            data["inverse"] = False
        else:
            data["inverse"] = True
    elif d == "fatter" or d == "skinnier":
        scale_factor = rand() + 1.0
        data["tform_data"]["lams"] = (scale_factor, 1.0, scale_factor)
        if d == "fatter":
            data["inverse"] = False
        else:
            data["inverse"] = True
    elif d == "taller" or d == "shorter":
        scale_factor = rand() + 1.0
        data["tform_data"]["lams"] = (1.0, scale_factor, 1.0)
        if d == "taller":
            data["inverse"] = False
        else:
            data["inverse"] = True
    elif d == "bigger" or d == "smaller":
        scale_factor = rand() + 1.0
        data["tform_data"]["lams"] = (scale_factor, scale_factor, scale_factor)
        if d == "bigger":
            data["inverse"] = False
        else:
            data["inverse"] = True
    else:
        print(d)
        raise Exception("what?")
    tform = shape_transforms.scale_sparse
    return tform, text, data


def rotate(schematic):
    data = {}
    angle = random.choice([90, -90])
    data["tform_data"] = {"angle": angle}
    data["inverse"] = False
    if angle == 90:
        ccw = "clockwise"
    else:
        ccw = "counter-clockwise"
    text = "rotate it " + ccw
    tform = shape_transforms.rotate
    return tform, text, data


def replace_by_block(schematic):
    data = {}
    data["inverse"] = False

    new_color = None
    # FIXME prob
    if rand() > 0.5:
        new_color = random.choice(list(COLOR_BID_MAP.keys()))
        new_idm = random.choice(COLOR_BID_MAP[new_color])
    else:
        new_idm = random.choice(NEW_BLOCK_CHOICES)
    data["tform_data"] = {"new_idm": new_idm}

    if rand() > 0.25:
        idx = tuple(random.choice(np.transpose(schematic[:, :, :, 0].nonzero())))
        idm = tuple(schematic[idx].squeeze())
        if rand() > 0.5 or not (BID_TO_COLOR.get(idm)):
            block_name = BLOCK_DATA["bid_to_name"][idm]
        else:
            block_name = BID_TO_COLOR[idm] + " blocks"
        text = "change all the " + block_name + " to "
        data["tform_data"]["current_idm"] = idm
    else:
        data["tform_data"]["every_n"] = 1
        text = "change all the blocks to "

    if new_color:
        text = text + new_color + " blocks "
    else:
        text = text + BLOCK_DATA["bid_to_name"][new_idm]

    tform = shape_transforms.replace_by_blocktype
    return tform, text, data


# TODO middle...
# TODO look vector and sides + front + back
def replace_by_halfspace(schematic):
    data = {}
    data["inverse"] = False

    new_color = None
    # FIXME prob
    if rand() > 0.5:
        new_color = random.choice(list(COLOR_BID_MAP.keys()))
        new_idm = random.choice(COLOR_BID_MAP[new_color])
    else:
        new_idm = random.choice(NEW_BLOCK_CHOICES)
    data["tform_data"] = {"new_idm": new_idm}

    geometry = {}
    geometry["offset"] = np.array(schematic.shape[:3]) / 2 + 0.5
    text = "make the "
    if rand() > 0.5:
        nz = np.transpose(schematic[:, :, :, 0].nonzero())
        mins = np.min(nz, axis=0)
        maxs = np.max(nz, axis=0)
        amount = "quarter "
        geometry["threshold"] = (maxs[1] - mins[1]) / 4
    else:
        amount = "half "
        geometry["threshold"] = 0.0

    if rand() > 0.5:
        text = text + "top " + amount
        geometry["v"] = np.array((0, 1.0, 0))
    else:
        text = text + "bottom " + amount
        geometry["v"] = np.array((0, -1.0, 0))

    data["tform_data"]["geometry"] = geometry

    if new_color:
        text = text + new_color
    else:
        text = text + BLOCK_DATA["bid_to_name"][new_idm]

    tform = shape_transforms.replace_by_halfspace
    return tform, text, data


def fill(schematic):
    data = {}
    data["tform_data"] = {}
    if rand() > 0.5:
        data["inverse"] = False
        text = "fill it up"
    else:
        data["inverse"] = True
        text = "hollow it out"

    tform = shape_transforms.fill_flat

    return tform, text, data


def get_schematic():
    shape_name = random.choice(sh.SHAPE_NAMES)
    opts = sh.SHAPE_HELPERS[shape_name]()
    opts["bid"] = sh.bid()
    blocks = sh.SHAPE_FNS[shape_name](**opts)
    if len(blocks) == 0:
        import ipdb

        ipdb.set_trace()
    return blocks


class ModifyData(tds.Dataset):
    def __init__(self, opts, dictionary=None):
        self.opts = opts
        self.templates = {
            "thicker": thicker,
            "scale": scale,
            "rotate": rotate,
            "replace_by_block": replace_by_block,
            #                          'replace_by_n': replace_by_n,
            "replace_by_halfspace": replace_by_halfspace,
            "fill": fill,
        }
        self.debug = opts.debug
        self.stored = []
        self.template_sampler = torch.distributions.Categorical(
            torch.Tensor(list(opts.tform_weights.values()))
        )
        self.tform_names = list(opts.tform_weights.keys())
        self.nexamples = opts.nexamples
        self.sidelength = opts.sidelength
        self.allow_same = opts.allow_same
        self.words_length = opts.words_length
        self.max_meta = opts.max_meta
        self.dictionary = dictionary
        if self.dictionary:
            if type(self.dictionary) is str:
                with open(dictionary, "rb") as f:
                    self.dictionary = pickle.load(f)
            self.unkword = len(self.dictionary["w2i"])
            self.padword = len(self.dictionary["w2i"]) + 1

    # for debug...
    def print_text(self, word_tensor):
        # words is the tensor output of indexing the dataset
        words = ""
        for i in range(word_tensor.shape[0]):
            w = word_tensor[i].item()
            if w == self.padword:
                break
            else:
                words = words + self.dictionary["i2w"][w] + " "
        return words

    # TODO deal with reshifting back?
    def generate_task(self):
        size_fits = False
        while not size_fits:
            schematic, _ = blocks_list_to_npy(get_schematic(), xyz=True)
            if max(schematic.shape) < self.sidelength:
                size_fits = True
                schematic = shape_transforms.moment_at_center(schematic, self.sidelength)
                tform_name = self.tform_names[self.template_sampler.sample()]
                tform, text, task_data = self.templates[tform_name](schematic)
        return tform, text, task_data, schematic

    def maybe_words_to_tensor(self, text):
        words = text.split()
        if self.dictionary:
            wt = torch.LongTensor(self.words_length).fill_(self.padword)
            for i in range(len(words)):
                wt[i] = self.dictionary["w2i"].get(words[i], self.unkword)
            return wt
        else:
            return words

    def maybe_hash_idm(self, x):
        if self.max_meta > 0:
            return x[:, :, :, 1] + self.max_meta * x[:, :, :, 0]
        else:
            return x

    def __getitem__(self, index):
        if self.debug > 0:
            if len(self.stored) == 0:
                for i in range(self.debug):
                    self.stored.append(self._generate_item(0))
            return self.stored[index]
        else:
            return self._generate_item(0)

    def _generate_item(self, index):
        change = False
        size_fits = False
        while not change and not size_fits:
            tform, text, task_data, schematic = self.generate_task()
            new_schematic = tform(schematic, **task_data["tform_data"])
            if max(new_schematic.shape) <= self.sidelength:
                size_fits = True
                new_schematic = shape_transforms.moment_at_center(new_schematic, self.sidelength)
                diff = (schematic - new_schematic).any()
                if self.allow_same:
                    change = True
                    if not diff:
                        text = ["no", "change"]
                else:
                    if diff:
                        change = True
        w = self.maybe_words_to_tensor(text)
        schematic = self.maybe_hash_idm(torch.LongTensor(schematic))
        new_schematic = self.maybe_hash_idm(torch.LongTensor(new_schematic))
        if task_data["inverse"]:
            return w, new_schematic, schematic
        else:
            return w, schematic, new_schematic

    def __len__(self):
        if self.debug > 0:
            return self.debug
        else:
            return self.nexamples


if __name__ == "__main__":
    from voxel_models.plot_voxels import SchematicPlotter
    import visdom
    import argparse

    vis = visdom.Visdom(server="http://localhost")
    sp = SchematicPlotter(vis)
    parser = argparse.ArgumentParser()
    parser.add_argument("--nexamples", type=int, default=100, help="size of epoch")
    parser.add_argument("--sidelength", type=int, default=32, help="size of epoch")
    parser.add_argument(
        "--save_dict_file",
        default="/private/home/aszlam/junk/word_modify_word_ids.pk",
        help="where to save word dict",
    )
    parser.add_argument(
        "--examples_for_dict",
        type=int,
        default=-1,
        help="if bigger than 0, uses that many examples to build the word dict and save it",
    )
    parser.add_argument(
        "--words_length", type=int, default=12, help="sentence pad length.  FIXME?"
    )
    opts = parser.parse_args()

    opts.allow_same = False
    opts.max_meta = -1
    opts.tform_weights = {
        "thicker": 1.0,
        "scale": 1.0,
        "rotate": 1.0,
        "replace_by_block": 1.0,
        #                          'replace_by_n': 1.0,
        "replace_by_halfspace": 1.0,
        "fill": 1.0,
    }
    opts.debug = False
    M = ModifyData(opts)

    def sample_and_draw():
        text, old_schematic, new_schematic = M[0]
        sp.drawMatplot(old_schematic)
        sp.drawMatplot(new_schematic, title=" ".join(text))
        return text, old_schematic, new_schematic

    #    for i in range(100):
    #        print(i)
    #        text, old_schematic, new_schematic = M[0]
    #    for i in range(10):
    #        a, b, c = sample_and_draw()

    w2i = {}
    i2w = {}
    wcount = 0
    for i in range(opts.examples_for_dict):
        text, a, b = M[0]
        for w in text:
            if w2i.get(w) is None:
                w2i[w] = wcount
                i2w[wcount] = w
                wcount += 1
                print("new word! " + str(wcount) + " " + w)
    if len(w2i) > 0:
        with open(opts.save_dict_file, "wb") as f:
            pickle.dump({"w2i": w2i, "i2w": i2w}, f)
