"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# flake8: noqa
import argparse
from ast import arguments
import json
import math
import functools
import logging
import os
from tqdm import tqdm
import random
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler

from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_model import (
    build_model,
    load_model,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_parsing import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.decoder_with_loss import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.encoder_decoder import *
from droidlet.perception.semantic_parsing.nsp_transformer_model.caip_dataset import *
from droidlet.perception.semantic_parsing.load_and_check_datasets import get_ground_truth
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_caip import caip_collate
from droidlet.perception.semantic_parsing.utils.nsp_logger import NSPLogger


GT_QUERY_ACTIONS = get_ground_truth(
    False,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../",
        "droidlet/artifacts/datasets/ground_truth/",
    ),
)


class ModelEvaluator:
    """Wrapper Class around evaluating model"""

    def __init__(self, args):
        self.args = args
        self.evaluate_results_logger = NSPLogger(
            "evaluation_results.csv",
            ["accuracy", "text_span_accuracy", "inference_speed"],
        )

    def evaluate(self, model, dataset, tokenizer):
        """Evaluation loop
        Args:
            model: Decoder to be evaluated
            dataset: Training dataset
            tokenizer: Tokenizer for input
        Returns:
            Accuracy
        """
        train_sampler = SequentialSampler(dataset)
        model_collate_fn = functools.partial(
            caip_collate, tokenizer=tokenizer, tree_to_text=self.args.tree_to_text
        )
        dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            collate_fn=model_collate_fn,
        )
        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=True)

        # Totals are accumulated over all batches then divided by the number of iterations.
        tot_steps = 0
        # Accuracy of LM predictions/internal nodes
        tot_int_acc = 0.0
        tot_span_acc = 0.0
        tot_accu = 0.0
        text_span_tot_acc = 0.0
        tot_time_cost = 0.0
        # disable autograd to reduce memory usage
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator):
                batch_tensors = [
                    t.to(model.decoder.lm_head.predictions.decoder.weight.device)
                    for t in batch[:4]
                ]
                x, x_mask, y, y_mask = batch_tensors
                time_s = time.time()
                outputs = model(x, x_mask, y, y_mask, None, True)
                time_e = time.time()
                # compute accuracy and add hard examples
                lm_acc, sp_acc, text_span_acc, full_acc = compute_accuracy(outputs, y)
                # book-keeping
                # shapes of accuracies are [B]
                tot_int_acc += (
                    lm_acc.sum().item() / lm_acc.shape[0]
                )  # internal_nodes_accuracy / batch_size
                tot_span_acc += (
                    sp_acc.sum().item() / sp_acc.shape[0]
                )  # weighted_accuracy / batch_size
                tot_accu += full_acc.sum().item() / full_acc.shape[0]
                # time cost
                tot_time_cost += (time_e - time_s) / full_acc.shape[0]
                tot_steps += 1
                # text span stats
                text_span_tot_acc += (
                    text_span_acc.sum().item() / text_span_acc.shape[0]
                )  # text_span_accuracy / batch_size

                if step % self.args.vis_step_size == 0 and self.args.show_samples:
                    show_examples(self.args, model, dataset, tokenizer)

        self.evaluate_results_logger.log_dialogue_outputs(
            [tot_accu / tot_steps, text_span_tot_acc / tot_steps, tot_steps / tot_time_cost]
        )

        logging.info("Accuracy: {:.3f}".format(tot_accu / tot_steps))
        logging.info("Text span accuracy: {:.3f}".format(text_span_tot_acc / tot_steps))
        logging.info("Inference speed (fps): {:.1f}".format(tot_steps / tot_time_cost))
        print("Evaluation done!")
        print("Accuracy: {:.3f}".format(tot_accu / tot_steps))
        print("Text span accuracy: {:.3f}".format(text_span_tot_acc / tot_steps))
        print(("Inference speed (fps): {:.1f}".format(tot_steps / tot_time_cost)))


def build_grammar(args):
    data = {"train": {}, "valid": {}, "test": {}}
    dtypes = list(args.dtype_samples.keys())
    for spl in data:
        for dt in dtypes:
            fname = pjoin(args.data_dir, "{}/{}.txt".format(spl, dt))
            logging.info("loading file {}".format(fname))
            if isfile(fname):
                data[spl][fname.split("/")[-1][:-4]] = process_txt_data(filepath=fname)
    full_tree, tree_i2w = make_full_tree(
        [(d_list, 1.0) for spl, dtype_dict in data.items() for dtype, d_list in dtype_dict.items()]
    )
    json.dump((full_tree, tree_i2w), open(args.tree_voc_file, "w"))


def show_examples(args, model, dataset, tokenizer, n=10):
    with torch.no_grad():
        for _ in range(n):
            cid = random.randint(0, len(dataset) - 1)
            chat = dataset[cid][2][1]
            btr = beam_search(
                chat, model, tokenizer, dataset, args.beam_size, args.well_formed_pen
            )
            if (
                btr[0][0].get("dialogue_type", "NONE") == "NOOP"
                and math.exp(btr[0][1]) < args.noop_thres
            ):
                tree = btr[1][0]
            else:
                tree = btr[0][0]
            print(chat)
            print(tree)
            print("*********************************")


def argument_parse(input_arg):
    """
    Parse input arguments of configuring model and dataset.
    Args:
        input_arg: a string consists of arguments separated by space.
    Returns:
        args: input arguments
    """

    parser = argparse.ArgumentParser()
    # dataset arguments
    parser.add_argument(
        "--data_dir",
        default="droidlet/artifacts/datasets/annotated_data/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--output_dir",
        default="droidlet/artifacts/models/nlu/ttad_bert_updated/",
        type=str,
        help="Where we save the model",
    )
    parser.add_argument(
        "--ground_truth_data_dir",
        default="droidlet/artifacts/datasets/ground_truth/",
        type=str,
        help="templated/templated_filters/annotated/short_commands/high_pri_commands data",
    )
    parser.add_argument(
        "--tree_voc_file",
        default="droidlet/artifacts/models/nlu/ttad_bert_updated/caip_test_model_tree.json",
        type=str,
        help="Pre-computed grammar and output vocabulary",
    )
    parser.add_argument("--tree_to_text", action="store_true", help="Back translation flag")
    # model arguments
    parser.add_argument("--model_name", default="caip_parser", type=str, help="Model name")
    parser.add_argument(
        "--pretrained_encoder_name",
        default="distilbert-base-uncased",
        type=str,
        help="Pretrained text encoder "
        "See full list at https://huggingface.co/transformers/pretrained_models.html",
    )
    parser.add_argument(
        "--decoder_config_name",
        default="bert-base-uncased",
        type=str,
        help="Name of Huggingface config used to initialize decoder architecture"
        "See full list at https://huggingface.co/transformers/pretrained_models.html",
    )
    parser.add_argument(
        "--num_decoder_layers",
        default=6,
        type=int,
        help="Number of transformer layers in the decoder",
    )
    parser.add_argument(
        "--num_highway", default=2, type=int, help="Number of highway layers in the mapping model"
    )
    parser.add_argument(
        "--model_dir",
        default="droidlet/artifacts/models/nlu/ttad_bert_updated/",
        type=str,
        help="Directory to pretrained NLU model",
    )
    # optimization arugments
    parser.add_argument("--batch_size", default=28, type=int, help="Batch size")
    parser.add_argument(
        "--train_encoder", default=1, type=int, help="Whether to finetune the encoder"
    )
    parser.add_argument(
        "--lambda_span_loss",
        default=0.5,
        type=float,
        help="Weighting between node and span prediction losses",
    )
    parser.add_argument(
        "--node_label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing for node prediction",
    )
    parser.add_argument(
        "--dtype_samples",
        default="annotated:1.0",
        type=str,
        help="Sampling probabilities for handling different data types",
    )
    parser.add_argument(
        "--examples_per_epoch", default=-1, type=int, help="Number of training examples per epoch"
    )
    parser.add_argument(
        "--rephrase_proba", default=-1.0, type=float, help="Only specify probablility of rephrases"
    )
    parser.add_argument(
        "--show_samples", action="store_true", help="show samples every few iterations"
    )
    # debug arguments
    parser.add_argument(
        "--vis_step_size",
        default=10,
        type=int,
        help="The number of iterations to visualize chat and parsed tree",
    )
    parser.add_argument(
        "--noop_thres", default=0.95, type=float, help="The threshold of NOOP action"
    )
    parser.add_argument(
        "--beam_size", default=5, type=int, help="Number of branches to keep in beam search"
    )
    parser.add_argument(
        "--well_formed_pen", default=1e2, type=float, help="Penalization for poorly formed trees"
    )
    parser.add_argument(
        "--load_ground_truth",
        action="store_false",
        help="Load ground truth for querying input chat",
    )

    if input_arg:
        args = parser.parse_args(input_arg.split())
    else:
        args = parser.parse_args()

    dtype_samples = {}
    for x in args.dtype_samples.split(";"):
        y = x.split(":")
        dtype_samples[y[0]] = float(y[1])
    args.dtype_samples = dtype_samples

    os.makedirs(args.output_dir, exist_ok=True)
    # HACK: allows us to give rephrase proba only instead of full dictionary
    if args.rephrase_proba > 0:
        args.dtype_samples = json.dumps(
            [["templated", 1.0 - args.rephrase_proba], ["rephrases", args.rephrase_proba]]
        )

    return args


def model_configure(args):
    """
    Configurate NLU model based on input arguments
    Args:
        args: input arguments of model and dataset configuration
    Returns:
        encoder_decoder:model class with pretrained model
        tokenizer: pretrained tokenizer
    """
    with open(args.tree_voc_file) as fd:
        full_tree, tree_i2w = json.load(fd)
    full_tree_voc = (full_tree, tree_i2w)

    logging.info("====== Loading Pretrained Parameters ======")
    sd, _, _, _, _ = load_model(args.model_dir)
    logging.info("====== Setting up Model ======")
    _, encoder_decoder, tokenizer = build_model(args, full_tree_voc[1])
    encoder_decoder.load_state_dict(sd, strict=True)
    encoder_decoder = encoder_decoder.cuda()
    encoder_decoder.eval()

    return encoder_decoder, tokenizer


def dataset_configure(args, tokenizer):
    """
    Configurate CAIP dataset based on input arguments
    Args:
        args: input arguments of model and dataset configuration
        tokenizer: pretrained tokenizer
    Returns:
        dataset:
    """
    with open(args.tree_voc_file) as fd:
        full_tree, tree_i2w = json.load(fd)
    full_tree_voc = (full_tree, tree_i2w)

    dataset = CAIPDataset(
        tokenizer,
        args,
        prefix="test",
        full_tree_voc=full_tree_voc,
        dtype="annotated",
    )

    return dataset


def query_model(chat, args, model, tokenizer, dataset):
    """
    Query mode for NLU model, which takes a sentence of natural language as input
    and outputs its logical form
    Args:
        chat (str): chat input
        args: input arguments of model and dataset configuration
        model: model class with pretrained model
        tokenizer: pretrained tokenizer
        dataset: caip dataset
    Returns:
        logical form (dict)
    """
    if args.load_ground_truth and chat in GT_QUERY_ACTIONS:
        tree = GT_QUERY_ACTIONS[chat]
    else:
        btr = beam_search(chat, model, tokenizer, dataset, args.beam_size, args.well_formed_pen)

        if (
            btr[0][0].get("dialogue_type", "NONE") == "NOOP"
            and math.exp(btr[0][1]) < args.noop_thres
        ):
            tree = btr[1][0]
        else:
            tree = btr[0][0]

    return tree


def eval_model(args, model, tokenizer, dataset):
    """
    Evaluation mode for NLU model, which computes the accuracy of the given dataset
    Args:
        args: input arguments of model and dataset configuration
        model: model class with pretrained model
        tokenizer: pretrained tokenizer
        dataset: caip dataset
    Returns:
    """
    model_evaluator = ModelEvaluator(args)
    model_evaluator.evaluate(model, dataset, tokenizer)


if __name__ == "__main__":
    args = argument_parse("")
    # TODO: print model hash
    print("loading model")
    model, tokenizer = model_configure(args)
    # TODO: print data hash
    print("loading dataset")
    dataset = dataset_configure(args, tokenizer)

    def get_parse(chat):
        return query_model(chat, args, model, tokenizer, dataset)
