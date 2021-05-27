"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import numpy as np
import re
from os.path import isfile, isdir
from os.path import join as pjoin
from torch.utils.data import Dataset
from .tokenization_utils import fixed_span_values
from .utils_caip import *


class CAIPDataset(Dataset):
    """Torch Dataset for the CAIP format, applies BPE and linearizes trees on-the-fly

    CAIP: CraftAssist Instruction Parsing

    Args:
        tokenizer: pre-trained tokenizer for input
        sampling: Whether to sample hard examples
        word_noise: Word noise
        sample_probas: Sampling probabilities for different data types
        dataset_length: Size of dataset
        tree_voc: Tree vocabulary file
        tree_idxs: Tree dictionary
        data: Dictionary containing datasets loaded into memory.

    """

    def __init__(
        self,
        tokenizer,
        args,
        prefix="train",
        dtype="templated",
        sampling=False,
        word_noise=0.0,
        full_tree_voc=None,
    ):
        assert isdir(args.data_dir)
        self.tokenizer = tokenizer
        self.tree_to_text = args.tree_to_text

        # We load the (input, tree) pairs for all data types and
        # initialize the hard examples buffer
        self.data = {"hard": []}
        self.sampling = sampling
        self.word_noise = word_noise
        dtype_samples = json.loads(args.dtype_samples)
        self.dtype = dtype
        self.dtypes = [t for t, p in dtype_samples]
        self.sample_probas = np.array([p for t, p in dtype_samples])
        self.sample_probas /= self.sample_probas.sum()
        if prefix == "train":
            for k in self.dtypes:
                fname = pjoin(args.data_dir, prefix, k + ".txt")
                print(fname)
                assert isfile(fname)
                print("loading {}".format(fname))
                self.data[k] = process_txt_data(fname)
            self.hard_buffer_size = 1024
            self.hard_buffer_counter = 0
        elif prefix == "":
            self.data[dtype] = []
        else:
            fname = pjoin(args.data_dir, prefix, dtype + ".txt")
            if isfile(fname):
                print("loading {}".format(fname))
                self.data[dtype] = process_txt_data(fname)
            else:
                print("could not find dataset {}".format(fname))
                self.data[dtype] = []

        # load meta-tree and tree vocabulary
        if full_tree_voc is None:
            print("making tree")
            ftr, tr_i2w = make_full_tree(
                [
                    (self.data["humanbot"], 3e5),
                    (self.data["prompts"], 1e5),
                    (self.data["templated"][:100000], 1),
                ]
            )
            self.full_tree = ftr
        else:
            full_tree, tr_i2w = full_tree_voc
            self.full_tree = full_tree
        spec_tokens = ["[PAD]", "unused", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "</S>"]
        self.tree_voc = spec_tokens[:] + tr_i2w + list(fixed_span_values)
        self.tree_idxs = dict([(w, i) for i, w in enumerate(self.tree_voc)])

        self.dataset_length = max([len(v) for v in self.data.values()])
        if args.examples_per_epoch > 0:
            self.dataset_length = min(self.dataset_length, args.examples_per_epoch)

    def _contains_span_indices(self, token_idx_list: list):
        return token_idx_list[1] >= 0 and token_idx_list[2] >= 0

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """Sample data type and get example"""
        if self.sampling:
            dtype = np.random.choice(self.dtypes, p=self.sample_probas)
            if len(self.data[dtype]) == 0:
                dtype = self.dtype
        else:
            dtype = self.dtype
        try:
            t = self.data[dtype][idx % len(self.data[dtype])]
            p_text, p_tree = t
        except Exception as e:
            print(e)
        text, tree = tokenize_linearize(
            p_text, p_tree, self.tokenizer, self.full_tree, self.word_noise
        )
        text_idx_ls = [self.tokenizer._convert_token_to_id(w) for w in text.split()]
        tree_idx_ls = [
            [
                self.tree_idxs[w],
                bi,
                ei,
                text_span_start,
                text_span_end,
                (
                    self.tree_idxs[fixed_span_val]
                    if type(fixed_span_val) == str
                    else fixed_span_val
                ),
            ]
            for w, bi, ei, text_span_start, text_span_end, fixed_span_val in [
                ("<S>", -1, -1, -1, -1, -1)
            ]
            + tree
            + [("</S>", -1, -1, -1, -1, -1)]
        ]
        if self.tree_to_text:
            stripped_tree_tokens = []
            for w, bi, ei in tree:
                tree_node = w.lower()
                tree_node_processed = re.sub("[^0-9a-zA-Z]+", " ", tree_node)
                tree_tokens = tree_node_processed.split(" ")
                stripped_tree_tokens += [x for x in tree_tokens if x != ""]

            extended_tree = ["[CLS]"] + stripped_tree_tokens + ["[SEP]"]
            tree_idx_ls = [self.tokenizer._convert_token_to_id(w) for w in extended_tree]
        return (text_idx_ls, tree_idx_ls, (text, p_text, p_tree))

    def add_hard_example(self, exple):
        """Add a given example for resampling."""
        if self.hard_buffer_counter < self.hard_buffer_size:
            self.data["hard"] += [exple]
        else:
            self.data["hard"][self.hard_buffer_counter % self.hard_buffer_size] = exple
        self.hard_buffer_counter += 1
