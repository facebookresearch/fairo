"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# flake8: noqa

import json
import pickle
import torch
import argparse
from transformers import AutoModel, AutoTokenizer, BertConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="craftassist/agent/datasets/annotated_data/",
        type=str,
        help="train/valid/test data",
    )
    args = parser.parse_args()

    model = "craftassist/agent/models/nlu/ttad_bert_updated/caip_test_model.pth"
    args_path = (
        "craftassist/agent/models/nlu/ttad_bert_updated/caip_test_model_args.pk"
    )
    args = pickle.load(open(args_path, "rb"))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
    with open(args.tree_voc_file) as fd:
        full_tree, tree_i2w = json.load(fd)
    full_tree_voc = (full_tree, tree_i2w)
    dataset = CAIPDataset(tokenizer, args, prefix="", full_tree_voc=(full_tree, tree_i2w))

    enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_config.vocab_size = len(tree_i2w) + 8
    bert_config.num_hidden_layers = args.num_decoder_layers
    dec_with_loss = DecoderWithLoss(bert_config, args, tokenizer)
    encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    encoder_decoder.load_state_dict(torch.load(model, map_location=map_location), strict=False)
    encoder_decoder = encoder_decoder.cuda()
    _ = encoder_decoder.eval()

    model_trainer = ModelTrainer(args)
    model_trainer.eval_model_on_dataset(
        encoder_decoder, "annotated", full_tree_voc, tokenizer, split="test"
    )


if __name__ == "__main__":
    main()
