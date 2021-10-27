"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import functools
import json
import logging
import logging.handlers
import os
import pickle
import math
from time import time

from os.path import isfile
from os.path import join as pjoin
from tqdm import tqdm

import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_model import build_model
from droidlet.perception.semantic_parsing.utils.nsp_logger import NSPLogger
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_parsing import (
    compute_accuracy,
    beam_search,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.utils_caip import (
    make_full_tree,
    process_txt_data,
    caip_collate,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.decoder_with_loss import (
    DecoderWithLoss,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.encoder_decoder import (
    EncoderDecoderWithLoss,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.optimizer_warmup import (
    OptimWarmupEncoderDecoder,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.caip_dataset import CAIPDataset


def save_model(model, model_identifier, dataset, args, full_tree_voc, epoch):
    M = {
        "state_dict": model.state_dict(),
        "tree_voc": dataset.tree_voc,
        "tree_idxs": dataset.tree_idxs,
        "full_tree_voc": full_tree_voc,
        "args": args,
    }
    path = pjoin(args.output_dir, "{}_ep{}.pth".format(model_identifier, epoch))
    print("saving model to PATH::{} at epoch {}".format(path, epoch))
    torch.save(M, path)


def show_examples(model, dataset, tokenizer, n=10):
    model.eval()
    with torch.no_grad():
        for cid in range(n):
            chat = dataset[cid][2][1]
            btr = beam_search(chat, model, tokenizer, dataset, 5, 10)
            if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < 0.95:
                tree = btr[1][0]
            else:
                tree = btr[0][0]
            print(chat)
            print(tree)
            print("*********************************")
    model.train()


class ModelTrainer:
    """Wrapper Class around training model and data loader"""

    def __init__(self, args):
        self.args = args
        # Initialize logger for machine-readable logs
        self.train_outputs_logger = NSPLogger(
            "training_outputs.csv",
            [
                "epoch",
                "iteration",
                "loss",
                "accuracy",
                "text_span_loss",
                "text_span_accuracy",
                "time",
            ],
        )
        self.valid_outputs_logger = NSPLogger(
            "valid_outputs.csv",
            [
                "epoch",
                "data_type",
                "loss",
                "accuracy",
                "text_span_loss",
                "text_span_accuracy",
                "time",
            ],
        )
        self.tensorboard_dir = args.tensorboard_dir

    def train(self, model, dataset, tokenizer, model_identifier, full_tree_voc):
        """Training loop (all epochs at once)

        Args:
            model: Decoder to be trained
            dataset: Training dataset
            tokenizer: Tokenizer for input
            model_identifier: Identifier string used when saving the model files
            full_tree_voc: full tree vocabulary

        Returns:
            Tuple of (Loss, Accuracy)

        """
        if self.tensorboard_dir:
            tensorboard_dir = os.path.join(self.tensorboard_dir, model_identifier)
            #            os.mkdir(self.tensorboard_dir, exist_ok=True)
            tb = torch.utils.tensorboard.SummaryWriter(log_dir=tensorboard_dir)
        else:
            tb = None

        # make data sampler
        train_sampler = RandomSampler(dataset)
        logging.info("Initializing train data sampler: {}".format(train_sampler))
        model_collate_fn = functools.partial(
            caip_collate, tokenizer=tokenizer, tree_to_text=self.args.tree_to_text
        )
        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            collate_fn=model_collate_fn,
        )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        # make optimizer
        optimizer = OptimWarmupEncoderDecoder(model, self.args)
        text_span_optimizer = Adam(
            [
                {"params": model.decoder.text_span_start_head.parameters()},
                {"params": model.decoder.text_span_end_head.parameters()},
            ],
            lr=0.001,
        )
        fixed_span_optimizer = Adam(
            [{"params": model.decoder.fixed_span_head.parameters()}], lr=0.001
        )
        text_span_loss_attenuation_factor = self.args.alpha
        fixed_value_loss_attenuation_factor = self.args.fixed_value_weight
        # training loop
        tot_steps = 0
        tot_loss = 0.0
        text_span_tot_loss = 0.0
        text_span_loc_loss = 0.0
        tot_accuracy = 0.0
        for e in range(self.args.num_epochs):
            logging.info("Epoch: {}".format(e))
            loc_steps = 0
            loc_loss = 0.0
            loc_int_acc = 0.0
            loc_span_acc = 0.0
            loc_full_acc = 0.0
            text_span_accuracy = 0.0
            st_time = time()
            for step, batch in enumerate(epoch_iterator):
                batch_examples = batch[-1]
                batch_tensors = [
                    t.to(model.decoder.lm_head.predictions.decoder.weight.device)
                    for t in batch[:4]
                ]
                x, x_mask, y, y_mask = batch_tensors
                if self.args.tree_to_text:
                    outputs = model(y, y_mask, x, x_mask)
                else:
                    outputs = model(x, x_mask, y, y_mask)
                loss = outputs["loss"]
                text_span_loss = outputs["text_span_loss"]
                fixed_span_loss = outputs["fixed_span_loss"]
                model.zero_grad()
                # backprop
                # Use separate optimizers for text span and fixed span heads
                text_span_loss.backward(retain_graph=True)
                text_span_optimizer.step()
                fixed_span_loss.backward(retain_graph=True)
                fixed_span_optimizer.step()

                loss.backward()
                # Add text span loss gradients
                model.decoder.bert_final_layer_out.grad = model.decoder.bert_final_layer_out.grad.add(
                    text_span_loss_attenuation_factor
                    * (
                        model.decoder.text_span_start_hidden_z.grad
                        + model.decoder.text_span_end_hidden_z.grad
                    )
                )
                # Add fixed value loss gradients
                model.decoder.bert_final_layer_out.grad = model.decoder.bert_final_layer_out.grad.add(
                    fixed_value_loss_attenuation_factor * (model.decoder.fixed_span_hidden_z.grad)
                )
                if step % self.args.param_update_freq == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                # compute accuracy and add hard examples
                if self.args.tree_to_text:
                    full_acc = compute_accuracy(outputs, x)
                    # hacky
                    lm_acc = full_acc
                else:
                    lm_acc, sp_acc, text_span_acc, full_acc = compute_accuracy(outputs, y)
                if self.args.hard:
                    if e > 0 or tot_steps > 2 * self.args.decoder_warmup_steps:
                        for acc, exple in zip(lm_acc, batch_examples):
                            if not acc.item():
                                if step % 400 == 100:
                                    print("ADDING HE:", step, exple[0])
                                dataset.add_hard_example(exple)
                # book-keeping
                # shapes of accuracies are [B]
                loc_int_acc += (
                    lm_acc.sum().item() / lm_acc.shape[0]
                )  # internal_nodes_accuracy / batch_size
                loc_full_acc += (
                    full_acc.sum().item() / full_acc.shape[0]
                )  # weighted_accuracy / batch_size
                tot_accuracy += full_acc.sum().item() / full_acc.shape[0]
                text_span_accuracy += (
                    text_span_acc.sum().item() / text_span_acc.shape[0]
                )  # text_span_accuracy / batch_size
                if not self.args.tree_to_text:
                    loc_span_acc += sp_acc.sum().item() / sp_acc.shape[0]
                loc_loss += loss.item()
                loc_steps += 1
                tot_loss += loss.item()
                tot_steps += 1
                text_span_tot_loss += text_span_loss.item()
                text_span_loc_loss += text_span_loss.item()
                if step % 400 == 0:
                    if args.show_samples:
                        show_examples(model, dataset, tokenizer)
                    if tb:
                        tb.add_scalar("accuracy", loc_full_acc / loc_steps, global_step=tot_steps)
                        tb.add_scalar("loss", loc_loss / loc_steps, global_step=tot_steps)
                    print(
                        "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                            e,
                            step,
                            loc_loss / loc_steps,
                            loc_full_acc / loc_steps,
                            time() - st_time,
                        )
                    )
                    logging.info(
                        "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                            e,
                            step,
                            loc_loss / loc_steps,
                            loc_full_acc / loc_steps,
                            time() - st_time,
                        )
                    )
                    logging.info("text span acc: {:.3f}".format(text_span_accuracy / loc_steps))
                    logging.info("text span loss: {:.3f}".format(text_span_loc_loss / loc_steps))
                    # Log training outputs to CSV
                    self.train_outputs_logger.log_dialogue_outputs(
                        [
                            e,
                            step,
                            loc_loss
                            / loc_steps,  # loss averaged over number of steps between gradient updates
                            loc_full_acc / loc_steps,
                            text_span_accuracy / loc_steps,
                            text_span_loc_loss / loc_steps,
                            time() - st_time,
                        ]
                    )
                    # Local calculations are reset each iteration (depends on frequency of updates)
                    loc_loss = 0
                    loc_steps = 0
                    loc_int_acc = 0.0
                    loc_span_acc = 0.0
                    loc_full_acc = 0.0
                    text_span_accuracy = 0.0
                    text_span_loc_loss = 0.0
            save_model(model, model_identifier, dataset, self.args, full_tree_voc, e)
            # Evaluating model
            model.eval()
            logging.info("evaluating model")
            for dtype, ratio in self.args.dtype_samples.items():
                l, a = self.eval_model_on_dataset(e, model, dtype, full_tree_voc, tokenizer)
                logging.info(
                    "evaluating on {} valid: \t Loss: {:.4f} \t Accuracy: {:.4f} at epoch {}".format(
                        dtype, l, a, e
                    )
                )
                print(
                    "evaluating on {} valid: \t Loss: {:.4f} \t Accuracy: {:.4f} at epoch {}".format(
                        dtype, l, a, e
                    )
                )
                if tb:
                    tb.add_scalar("val_accuracy_" + str(dtype), a, global_step=e)
                    tb.add_scalar("val_loss_" + str(dtype), l, global_step=e)

        return (tot_loss / tot_steps, tot_accuracy / tot_steps)

    def validate(self, model, dataset, tokenizer, args):
        """Validation: same as training loop but without back-propagation"""
        # make data sampler
        train_sampler = SequentialSampler(dataset)
        model_collate_fn = functools.partial(
            caip_collate, tokenizer=tokenizer, tree_to_text=args.tree_to_text
        )
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=model_collate_fn
        )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        # training loop
        # Totals are accumulated over all batches then divided by the number of iterations.
        tot_steps = 0
        # Loss is not normalized
        tot_loss = 0.0
        # Accuracy of LM predictions/internal nodes
        tot_int_acc = 0.0
        tot_span_acc = 0.0
        tot_accu = 0.0
        text_span_tot_acc = 0.0
        text_span_tot_loss = 0.0
        # disable autograd to reduce memory usage
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator):
                batch_tensors = [
                    t.to(model.decoder.lm_head.predictions.decoder.weight.device)
                    for t in batch[:4]
                ]
                x, x_mask, y, y_mask = batch_tensors
                outputs = model(x, x_mask, y, y_mask, None, True)
                loss = outputs["loss"]
                text_span_loss = outputs["text_span_loss"]
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
                tot_loss += loss.item()
                tot_steps += 1
                # text span stats
                text_span_tot_acc += (
                    text_span_acc.sum().item() / text_span_acc.shape[0]
                )  # text_span_accuracy / batch_size
                text_span_tot_loss += text_span_loss.item()
        return (
            tot_loss / tot_steps,
            tot_int_acc / tot_steps,
            tot_span_acc / tot_steps,
            tot_accu / tot_steps,
            text_span_tot_acc / tot_steps,
            text_span_tot_loss / tot_steps,
        )

    def eval_model_on_dataset(
        self, epoch, encoder_decoder, dtype, full_tree_voc, tokenizer, split="valid"
    ):
        """Evaluate model on a given validation dataset

        Args:
            encoder_decoder: model used for validation
            dtype: data type (name of file to load)
            full_tree_voc: full tree vocabulary
            tokenizer: pre-trained tokenizer for input

        """
        valid_dataset = CAIPDataset(
            tokenizer, self.args, prefix=split, dtype=dtype, full_tree_voc=full_tree_voc
        )
        l, _, _, a, text_span_acc, text_span_loss = self.validate(
            encoder_decoder, valid_dataset, tokenizer, self.args
        )
        # logging.info(
        #    "evaluating on {} valid: \t Loss: {:.4f} \t Accuracy: {:.4f}".format(dtype, l, a)
        # )
        logging.info(
            "text span Loss: {:.4f} \t Accuracy: {:.4f}".format(text_span_loss, text_span_acc)
        )
        self.valid_outputs_logger.log_dialogue_outputs(
            [epoch, dtype, l, a, text_span_acc, text_span_loss, time()]
        )
        return l, a


def generate_model_name(args, optional_identifier=""):
    """Generate a unique string identifier for the current run.

    Args:
        args: Parser arguments
        optional_identifier: Optional string appended to end of the model identifier

    Returns:
        String

    """
    name = ""
    # unix time in seconds, used as a unique identifier
    time_now = round(time())

    args_keys = {
        "batch_size": "batch",
        "decoder_learning_rate": "dec_lr",
        "decoder_warmup_steps": "dec_ws",
        "dtype_samples": "data",
        "encoder_learning_rate": "enc_lr",
        "encoder_warmup_steps": "enc_ws",
        "model_name": "name",
        "node_label_smoothing": "n_ls",
        "num_highway": "hw",
        "param_update_freq": "upd_frq",
        "word_dropout": "word_drp",
        "alpha": "a",
        "train_encoder": "tr",
        "fixed_value_weight": "fv",
    }
    dsets = {"templated": "t", "templated_filters": "tf", "annotated": "a"}
    for k, v in vars(args).items():
        if k in args_keys:
            if k == "dtype_samples":
                v = "_".join([dsets[k] + str(v) for k, v in args.dtype_samples.items()])
            name += "{param}{value}-".format(param=args_keys[k], value=v)
    # In case we want additional identification for the model, eg. test run
    name += "{time}|".format(time=time_now)
    name += optional_identifier
    return name


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="agents/craftassist/datasets/annotated_data/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument("--tensorboard_dir", default="")
    parser.add_argument(
        "--output_dir",
        default="agents/craftassist/models/nlu/ttad_bert_updated/",
        type=str,
        help="Where we save the model",
    )
    parser.add_argument("--model_name", default="caip_parser", type=str, help="Model name")
    parser.add_argument(
        "--tree_voc_file",
        default="agents/craftassist/models/nlu/ttad_bert_updated/caip_test_model_tree.json",
        type=str,
        help="Pre-computed grammar and output vocabulary",
    )
    # model arguments
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
    # optimization arguments
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="Optimizer in [adam|adagrad]"
    )
    parser.add_argument("--batch_size", default=56, type=int, help="Batch size")
    parser.add_argument("--param_update_freq", default=1, type=int, help="Group N batch updates")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument(
        "--examples_per_epoch", default=-1, type=int, help="Number of training examples per epoch"
    )
    parser.add_argument(
        "--train_encoder", default=1, type=int, help="Whether to finetune the encoder"
    )
    parser.add_argument(
        "--encoder_warmup_steps",
        default=1,
        type=int,
        help="Learning rate warmup steps for the encoder",
    )
    parser.add_argument(
        "--encoder_learning_rate", default=0.0, type=float, help="Learning rate for the encoder"
    )
    parser.add_argument(
        "--decoder_warmup_steps",
        default=1000,
        type=int,
        help="Learning rate warmup steps for the decoder",
    )
    parser.add_argument(
        "--decoder_learning_rate", default=1e-5, type=float, help="Learning rate for the decoder"
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
        "--span_label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing for span prediction",
    )
    parser.add_argument(
        "--dtype_samples",
        default="templated:.55;templated_filters:.05;annotated:.4",
        type=str,
        help="Sampling probabilities for handling different data types",
    )
    parser.add_argument(
        "--rephrase_proba", default=-1.0, type=float, help="Only specify probablility of rephrases"
    )
    parser.add_argument(
        "--word_dropout",
        default=0.0,
        type=float,
        help="Probability of replacing input token with [UNK]",
    )
    parser.add_argument(
        "--encoder_dropout", default=0.0, type=float, help="Apply dropout to encoder output"
    )
    parser.add_argument(
        "--show_samples", action="store_true", help="show samples every few iterations"
    )
    parser.add_argument("--tree_to_text", action="store_true", help="Back translation flag")
    parser.add_argument(
        "--optional_identifier", default="", type=str, help="Optional run info eg. debug or test"
    )
    parser.add_argument(
        "--hard", default=1, type=int, help="Whether to feed in failed examples during training"
    )
    parser.add_argument(
        "--alpha",
        default=0.8,
        type=float,
        help="Attenuation factor for text span loss gradient affecting shared layers for tree structure prediction",
    )
    parser.add_argument(
        "--fixed_value_weight",
        default=0.1,
        type=float,
        help="Attenuation factor for fixed value loss gradient affecting shared layers for tree structure prediction",
    )
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

    model_identifier = generate_model_name(args, args.optional_identifier)

    # set up logging
    l_handler = logging.handlers.WatchedFileHandler(
        "{}/{}.log".format(args.output_dir, model_identifier)
    )
    l_format = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    l_handler.setFormatter(l_format)
    l_root = logging.getLogger()
    l_root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    l_root.addHandler(l_handler)
    logging.info("****** Args ******")
    logging.info(vars(args))
    logging.info("model identifier: {}".format(model_identifier))
    if isfile(args.tree_voc_file):
        logging.info("====== Loading Grammar ======")
    else:
        logging.info("====== Making Grammar ======")
        build_grammar(args)
    with open(args.tree_voc_file) as fd:
        full_tree, tree_i2w = json.load(fd)

    logging.info("====== Setting up Model ======")
    dec_with_loss, encoder_decoder, tokenizer = build_model(args, tree_i2w)

    logging.info("====== Loading Dataset ======")
    train_dataset = CAIPDataset(
        tokenizer,
        args,
        prefix="train",
        sampling=True,
        word_noise=args.word_dropout,
        full_tree_voc=(full_tree, tree_i2w),
    )

    logging.info("====== Training Model ======")
    encoder_decoder = encoder_decoder.cuda()
    encoder_decoder.train()
    full_tree_voc = (full_tree, tree_i2w)
    model_trainer = ModelTrainer(args)
    loss, accu = model_trainer.train(
        encoder_decoder, train_dataset, tokenizer, model_identifier, full_tree_voc
    )
