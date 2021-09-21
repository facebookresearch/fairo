"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import os
import math
import pickle

import torch

from .utils_model import build_model, load_model
from .utils_parsing import beam_search
from .utils_parsing import *
from .decoder_with_loss import *
from .encoder_decoder import *
from .caip_dataset import *


class TTADBertModel(object):
    """
    TTAD model class that loads a pretrained model and runs inference in the agent.

    Attributes:
        tokenizer (str): Pretrained tokenizer used to tokenize input. Runs end-to-end
            tokenization, eg. split punctuation, BPE.
        dataset (CAIPDataset): CAIP (CraftAssist Instruction Parsing) Dataset. Note that
            this is empty during inference.
        encoder_decoder (EncoderDecoderWithLoss): Transformer model class. See

    Args:
        model_dir (str): Path to directory containing all files necessary to
            load and run the model, including args, tree mappings and the checkpointed model.
            Semantic parsing models used by current project are in ``ttad_bert_updated``.
            eg. semantic parsing model is ``ttad_bert_updated/caip_test_model.pth``
        data_dir (str): Path to directory containing all datasets used by the NSP model.
            Note that this data is not used in inference, rather we load from the ground truth
            data directory.
    """

    def __init__(self, model_dir, data_dir, model_name="caip_test_model"):
        sd, tree_voc, tree_idxs, args, full_tree_voc = load_model(model_dir)
        decoder_with_loss, encoder_decoder, tokenizer = build_model(args, full_tree_voc[1])
        args.data_dir = data_dir
        self.tokenizer = tokenizer
        self.dataset = CAIPDataset(self.tokenizer, args, prefix="", full_tree_voc=full_tree_voc)
        self.encoder_decoder = encoder_decoder
        self.encoder_decoder.load_state_dict(sd, strict=True)
        if torch.cuda.is_available():
            self.encoder_decoder.cuda()
        self.encoder_decoder.eval()

    def parse(self, chat, noop_thres=0.95, beam_size=5, well_formed_pen=1e2):
        """Given an incoming chat, query the parser and return a logical form.
        Uses beam search decoding, see :any:`beam_search`

        Args:
            chat (str): Preprocessed chat command from a player. Used as text input to parser.

        Returns:
            dict: Logical form.

        """
        btr = beam_search(
            chat, self.encoder_decoder, self.tokenizer, self.dataset, beam_size, well_formed_pen
        )
        if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
            tree = btr[1][0]
        else:
            tree = btr[0][0]
        return tree
