"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import json
import os
import math
import pickle

import torch

from transformers import AutoModel, AutoTokenizer, BertConfig

from droidlet.dialog.ttad.ttad_transformer_model.utils_parsing import *
from droidlet.dialog.ttad.ttad_transformer_model.decoder_with_loss import *
from droidlet.dialog.ttad.ttad_transformer_model.encoder_decoder import *
from droidlet.dialog.ttad.ttad_transformer_model.caip_dataset import *


class TTADBertModel(object):
    """TTAD model class that loads a pretrained model and runs inference in the agent.

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
        model_name = os.path.join(model_dir, model_name)
        with open(model_name + "_args.pk", "rb") as fd:
            args = pickle.load(fd)

        args.data_dir = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
        with open(model_name + "_tree.json") as fd:
            full_tree, tree_i2w = json.load(fd)
        self.dataset = CAIPDataset(
            self.tokenizer, args, prefix="", full_tree_voc=(full_tree, tree_i2w)
        )

        enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_config.is_decoder = True
        bert_config.add_cross_attention = True
        bert_config.vocab_size = len(tree_i2w) + 8

        bert_config.num_hidden_layers = args.num_decoder_layers
        dec_with_loss = DecoderWithLoss(bert_config, args, self.tokenizer)
        self.encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.encoder_decoder.load_state_dict(
            torch.load(model_name + ".pth", map_location=map_location), strict=False
        )
        self.encoder_decoder = (
            self.encoder_decoder.cuda()
            if torch.cuda.is_available()
            else self.encoder_decoder.cpu()
        )
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
