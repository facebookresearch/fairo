import os
import torch
import pickle
import json
from droidlet.perception.semantic_parsing.nsp_transformer_model.decoder_with_loss import (
    DecoderWithLoss,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.encoder_decoder import (
    EncoderDecoderWithLoss,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.tokenization_utils import TREE_KEYs
from transformers import AutoModel, AutoTokenizer, BertConfig, load_tf_weights_in_realm


def build_model(args, tree_i2w):
    # Initialize tokenier
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
    # Initialize bert configuration for decoder
    bert_config = BertConfig.from_pretrained(args.decoder_config_name)
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    if args.tree_to_text:
        tokenizer.add_tokens(tree_i2w)

    # add new tokens into tokenzier in case
    tree_keys = set(TREE_KEYs) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(tree_keys))
    # update word embedding for encoder
    enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
    enc_model.resize_token_embeddings(len(tokenizer))

    bert_config.vocab_size = len(tokenizer)
    dec_with_loss = DecoderWithLoss(bert_config, args, tokenizer)
    encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
    return dec_with_loss, encoder_decoder, tokenizer


def load_model(model_dir, model_name="caip_test_model"):
    path = os.path.join(model_dir, model_name + ".pth")
    try:
        M = torch.load(path, map_location="cpu")
        sd = M["state_dict"]
        tree_voc = M["tree_voc"]
        tree_idxs = M["tree_idxs"]
        args = M["args"]
        full_tree_voc = M["full_tree_voc"]
    except:
        try:
            print("WARNING: failed to load model, trying old-style model load")
            sd = torch.load(path, map_location="cpu")
            args_path = os.path.join(model_dir, model_name + "_args.pk")
            args = pickle.load(open(args_path, "rb"))
            tree_path = os.path.join(model_dir, model_name + "_tree.json")
            with open(tree_path) as fd:
                # with open(args.tree_voc_file) as fd:
                full_tree, tree_i2w = json.load(fd)
                full_tree_voc = (full_tree, tree_i2w)
        except:
            raise Exception("Failed to load model with path {}".format(path))

    return sd, tree_voc, tree_idxs, args, full_tree_voc
