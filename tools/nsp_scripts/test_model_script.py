"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# flake8: noqa

import json
import math
import pickle
import torch
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_model import (
    build_model,
    load_model,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_parsing import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.decoder_with_loss import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.encoder_decoder import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.caip_dataset import *

from pprint import pprint

map_location = None if torch.cuda.is_available() else torch.device("cpu")
model_dir = "agents/craftassist/models/nlu/ttad_bert_updated/"
sd, tree_voc, tree_idxs, args, full_tree_voc = load_model(model_dir)
decoder_with_loss, encoder_decoder, tokenizer = build_model(args, full_tree_voc[1])
encoder_decoder.load_state_dict(sd, strict=True)
encoder_decoder = encoder_decoder.cuda()
_ = encoder_decoder.eval()

dataset = CAIPDataset(tokenizer, args, prefix="", full_tree_voc=full_tree_voc)


def get_beam_tree(chat, noop_thres=0.95, beam_size=5, well_formed_pen=1e2):
    """Given a chat, runs beam search with the pretrained model and returns a logical form.

    Args:
        chat (str): text input
    Returns:
        logical_form (dict)
    """
    btr = beam_search(chat, encoder_decoder, tokenizer, dataset, beam_size, well_formed_pen)
    if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
        tree = btr[1][0]
    else:
        tree = btr[0][0]
    return tree
