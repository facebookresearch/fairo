"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# flake8: noqa

import json
import math
import pickle
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_model import build_model
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_parsing import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.decoder_with_loss import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.encoder_decoder import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.caip_dataset import *

from pprint import pprint

map_location = None if torch.cuda.is_available() else torch.device("cpu")
model = "craftassist/agent/models/semantic_parser/ttad_bert_updated/caip_test_model.pth"
try:
    M = torch.load(model, map_location=map_location)
    sd = M["state_dict"]
    tree_voc = M["tree_voc"]
    tree_idxs = M["tree_idxs"]
    args = M["args"]
    full_tree_voc = M["full_tree_voc"]
except:
    print("WARNING: failed to load model, trying old-style model load")
    sd = torch.load(model, map_location=map_location)
    args_path = (
        "craftassist/agent/models/semantic_parser/ttad_bert_updated/caip_test_model_args.pk"
    )
    args = pickle.load(open(args_path, "rb"))
    tree_path = (
        "craftassist/agent/models/semantic_parser/ttad_bert_updated/caip_test_model_tree.json"
    )
    with open(tree_path) as fd:
        # with open(args.tree_voc_file) as fd:
        full_tree, tree_i2w = json.load(fd)
        full_tree_voc = (full_tree, tree_i2w)


# tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
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
