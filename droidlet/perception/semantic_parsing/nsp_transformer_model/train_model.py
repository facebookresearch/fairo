"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import functools
import json
import logging
import logging.handlers
import os
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
from droidlet.perception.semantic_parsing.nsp_transformer_model.optimizer_warmup import (
    OptimWarmupEncoderDecoder,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.caip_dataset import CAIPDataset


class NLUModelTrainer:
    """Wrapper Class around NLU model trainer"""

    def __init__(self, args, model, tokenizer, model_identifier, full_tree_voc):
        # Setup arguments
        self.args = args
        self.text_span_loss_attenuation_factor = self.args.alpha
        self.fixed_value_loss_attenuation_factor = self.args.fixed_value_weight
        self.model_identifier = model_identifier
        self.full_tree_voc = full_tree_voc

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
        if self.args.tensorboard_dir:
            tensorboard_dir = os.path.join(self.args.tensorboard_dir, self.model_identifier)
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=tensorboard_dir)
        else:
            self.tb = None

        # Assign model with loss and tokenizer
        self.model = model
        self.tokenizer = tokenizer

        # Initialize different optimizers for text span and fixed span heads
        self.optimizer = OptimWarmupEncoderDecoder(model, self.args)
        self.text_span_optimizer = Adam(
            [
                {"params": model.decoder.text_span_start_head.parameters()},
                {"params": model.decoder.text_span_end_head.parameters()},
            ],
            lr=0.001,
        )
        self.fixed_span_optimizer = Adam(
            [{"params": model.decoder.fixed_span_head.parameters()}], lr=0.001
        )

    def run_epoch(self, phase, epoch, dataset, dataloader):
        """
        Model runs through given dataloader for single epoch
        Args:
            phase: Running phase, train or validation
            epoch: The current training epoch
            dataset: Dataset
            dataloader: Dataset loader
        Returns:
            Tuple of (Loss, Accuracy, Steps, Train Dataset)
            or
            (Loss, Int. Accuracy, Span Accuracy, Accuracy, Text Span Accuracy, Text Span Loss)
        """
        model = self.model
        if phase == "train":
            model.train()
        else:
            model.eval()

        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=True)
        ep_steps = 0
        ep_loss = 0.0
        ep_int_acc = 0.0
        ep_span_acc = 0.0
        ep_full_acc = 0.0
        ep_text_span_accuracy = 0.0
        ep_text_span_loss = 0.0

        loc_steps = 0
        loc_loss = 0.0
        loc_int_acc = 0.0
        loc_span_acc = 0.0
        loc_full_acc = 0.0
        loc_text_span_accuracy = 0.0
        loc_text_span_loss = 0.0
        st_time = time()
        for step, batch in enumerate(epoch_iterator):
            batch_examples = batch[-1]
            batch_tensors = [
                t.to(model.decoder.lm_head.predictions.decoder.weight.device) for t in batch[:4]
            ]
            x, x_mask, y, y_mask = batch_tensors
            # pass batch data through model
            if self.args.tree_to_text:
                outputs = model(y, y_mask, x, x_mask)
            else:
                outputs = model(x, x_mask, y, y_mask)

            loss = outputs["loss"]
            text_span_loss = outputs["text_span_loss"]
            fixed_span_loss = outputs["fixed_span_loss"]

            # parameter update for training phase only
            if phase == "train":
                # set gradients to zero
                model.zero_grad()

                # backprop
                # Use separate optimizers for text span and fixed span heads
                text_span_loss.backward(retain_graph=True)
                self.text_span_optimizer.step()
                fixed_span_loss.backward(retain_graph=True)
                self.fixed_span_optimizer.step()
                loss.backward()

                # Add text span loss gradients
                model.decoder.bert_final_layer_out.grad = (
                    model.decoder.bert_final_layer_out.grad.add(
                        self.text_span_loss_attenuation_factor
                        * (
                            model.decoder.text_span_start_hidden_z.grad
                            + model.decoder.text_span_end_hidden_z.grad
                        )
                    )
                )
                # Add fixed value loss gradients
                model.decoder.bert_final_layer_out.grad = (
                    model.decoder.bert_final_layer_out.grad.add(
                        self.fixed_value_loss_attenuation_factor
                        * (model.decoder.fixed_span_hidden_z.grad)
                    )
                )
                if step % self.args.param_update_freq == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizer.step()

            # compute accuracy and add hard examples
            if self.args.tree_to_text:
                full_acc = compute_accuracy(outputs, x)
                # hacky
                lm_acc = full_acc
            else:
                lm_acc, sp_acc, text_span_acc, full_acc = compute_accuracy(outputs, y)

            # add hard examples into the training dataset
            if phase == "train" and self.args.hard:
                if (
                    epoch > 0
                    or epoch * len(epoch_iterator) + ep_steps > 2 * self.args.decoder_warmup_steps
                ):
                    for acc, exple in zip(lm_acc, batch_examples):
                        if not acc.item():
                            if step % self.args.hard_iter == 100:
                                print("ADDING HE:", step, exple[0])
                            dataset.add_hard_example(exple)

            # book-keeping
            # shapes of accuracies are [B]
            # internal_nodes_accuracy / batch_size
            loc_int_acc += lm_acc.sum().item() / lm_acc.shape[0]
            ep_int_acc += lm_acc.sum().item() / lm_acc.shape[0]
            # weighted_accuracy / batch_size
            loc_full_acc += full_acc.sum().item() / full_acc.shape[0]
            ep_full_acc += full_acc.sum().item() / full_acc.shape[0]
            # text_span_accuracy / batch_size
            loc_text_span_accuracy += text_span_acc.sum().item() / text_span_acc.shape[0]
            if not self.args.tree_to_text:
                loc_span_acc += sp_acc.sum().item() / sp_acc.shape[0]
                ep_span_acc += sp_acc.sum().item() / sp_acc.shape[0]

            # total loss
            loc_loss += loss.item()
            ep_loss += loss.item()
            # number of iterations
            loc_steps += 1
            ep_steps += 1
            # test span loss
            loc_text_span_loss += text_span_loss.item()
            ep_text_span_loss += text_span_loss.item()

            if phase == "train" and step % self.args.log_iter == 0:
                if self.args.show_samples:
                    self.show_examples(dataset)
                if self.tb:
                    self.tb.add_scalar(
                        "accuracy",
                        loc_full_acc / loc_steps,
                        global_step=epoch * len(epoch_iterator) + ep_steps,
                    )
                    self.tb.add_scalar(
                        "loss",
                        loc_loss / loc_steps,
                        global_step=epoch * len(epoch_iterator) + ep_steps,
                    )
                print(
                    "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                        epoch,
                        step,
                        loc_loss / loc_steps,
                        loc_full_acc / loc_steps,
                        time() - st_time,
                    )
                )
                logging.info(
                    "{:2d} - {:5d} \t L: {:.3f} A: {:.3f} \t {:.2f}".format(
                        epoch,
                        step,
                        loc_loss / loc_steps,
                        loc_full_acc / loc_steps,
                        time() - st_time,
                    )
                )
                logging.info("text span acc: {:.3f}".format(loc_text_span_accuracy / loc_steps))
                logging.info("text span loss: {:.3f}".format(loc_text_span_loss / loc_steps))
                # Log training outputs to CSV
                self.train_outputs_logger.log_dialogue_outputs(
                    [
                        epoch,
                        step,
                        # loss averaged over number of steps between gradient updates
                        loc_loss / loc_steps,
                        loc_full_acc / loc_steps,
                        loc_text_span_accuracy / loc_steps,
                        loc_text_span_loss / loc_steps,
                        time() - st_time,
                    ]
                )
                # Local calculations are reset (depends on frequency of updates)
                loc_steps = 0
                loc_loss = 0.0
                loc_int_acc = 0.0
                loc_span_acc = 0.0
                loc_full_acc = 0.0
                loc_text_span_accuracy = 0.0
                loc_text_span_loss = 0.0

        if phase == "train":
            return (ep_loss, ep_full_acc, ep_steps, dataset)
        else:
            return (
                ep_loss / ep_steps,
                ep_int_acc / ep_steps,
                ep_span_acc / ep_steps,
                ep_full_acc / ep_steps,
                ep_text_span_accuracy / ep_steps,
                ep_text_span_loss / ep_steps,
            )

    def train(self, epoch, dataset):
        """Training loop (one epoch at once)
        Args:
            epoch: The current training epoch
            dataset: Training dataset
        Returns:
            Tuple of (Loss, Accuracy, Steps, Train Dataset)
        """
        # make data sampler
        train_sampler = RandomSampler(dataset)
        model_collate_fn = functools.partial(
            caip_collate, tokenizer=self.tokenizer, tree_to_text=self.args.tree_to_text
        )
        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            collate_fn=model_collate_fn,
        )

        return self.run_epoch("train", epoch, dataset, train_dataloader)

    def validate(self, epoch, dataset, dtype):
        """Validation loop
        Args:
            epoch: The current training epoch
            dataset: Validation dataset
            dtype: Type of data, [templated, templated_filters, annotated]
        Returns:
            Tuple(Loss, Accuracy)
        """
        # make data sampler
        val_sampler = SequentialSampler(dataset)
        model_collate_fn = functools.partial(
            caip_collate, tokenizer=self.tokenizer, tree_to_text=self.args.tree_to_text
        )
        val_dataloader = DataLoader(
            dataset,
            sampler=val_sampler,
            batch_size=self.args.batch_size,
            collate_fn=model_collate_fn,
        )

        print(len(val_dataloader))

        loss, _, _, acc, text_span_acc, text_span_loss = self.run_epoch(
            "val", epoch, dataset, val_dataloader
        )
        logging.info(
            "text span Loss: {:.4f} \t Accuracy: {:.4f}".format(text_span_loss, text_span_acc)
        )
        self.valid_outputs_logger.log_dialogue_outputs(
            [epoch, dtype, loss, acc, text_span_acc, text_span_loss, time()]
        )

        logging.info(
            "evaluating on {} valid: \t Loss: {:.4f} \t Accuracy: {:.4f} at epoch {}".format(
                dtype, loss, acc, epoch
            )
        )
        print(
            "evaluating on {} valid: \t Loss: {:.4f} \t Accuracy: {:.4f} at epoch {}".format(
                dtype, loss, acc, epoch
            )
        )
        if self.tb:
            self.tb.add_scalar("val_accuracy_" + str(dtype), acc, global_step=epoch)
            self.tb.add_scalar("val_loss_" + str(dtype), loss, global_step=epoch)

        return loss, acc

    def save_model(self, epoch, dataset):
        M = {
            "state_dict": self.model.state_dict(),
            "tree_voc": dataset.tree_voc,
            "tree_idxs": dataset.tree_idxs,
            "full_tree_voc": self.full_tree_voc,
            "args": self.args,
        }
        path = pjoin(self.args.output_dir, "{}_ep{}.pth".format(self.model_identifier, epoch))
        print("saving model to PATH::{} at epoch {}".format(path, epoch))
        torch.save(M, path)

    def show_examples(self, dataset):
        self.model.eval()
        with torch.no_grad():
            for cid in range(self.args.nm_shows):
                chat = dataset[cid][2][1]
                btr = beam_search(
                    chat,
                    self.model,
                    self.tokenizer,
                    self.args.beam_size,
                    self.args.well_formed_pen,
                )
                if (
                    btr[0][0].get("dialogue_type", "NONE") == "NOOP"
                    and math.exp(btr[0][1]) < self.args.noop_thres
                ):
                    tree = btr[1][0]
                else:
                    tree = btr[0][0]
                print(chat)
                print(tree)
                print("*********************************")
        self.model.train()


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
        default="droidlet/artifacts/datasets/annotated_data/",
        type=str,
        help="train/valid/test data",
    )
    parser.add_argument(
        "--root_dir",
        default="",
        type=str,
        help="The root folder of the fairo project",
    )
    parser.add_argument("--tensorboard_dir", default="")
    parser.add_argument(
        "--output_dir",
        default="droidlet/artifacts/models/nlu/ttad_bert_updated/",
        type=str,
        help="Where we save the model",
    )
    parser.add_argument("--model_name", default="caip_parser", type=str, help="Model name")
    parser.add_argument(
        "--tree_voc_file",
        default="droidlet/artifacts/models/nlu/ttad_bert_updated/caip_test_model_tree.json",
        type=str,
        help="Pre-computed grammar and output vocabulary",
    )
    parser.add_argument(
        "--hard_iter",
        default=400,
        type=int,
        help="Number of iterations to add hard examples",
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
    # debug parameters
    parser.add_argument(
        "--log_iter",
        default=400,
        type=int,
        help="The number of iteration for printing training progress and showing examples",
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
        "--nm_shows", default=10, type=int, help="Number of branches to keep in beam search"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    args = parser.parse_args()

    # parse proportion of different types of data samples from input arguments
    dtype_samples = {}
    for x in args.dtype_samples.split(";"):
        y = x.split(":")
        dtype_samples[y[0]] = float(y[1])
    args.dtype_samples = dtype_samples

    # HACK: allows us to give rephrase proba only instead of full dictionary
    if args.rephrase_proba > 0:
        args.dtype_samples = json.dumps(
            [["templated", 1.0 - args.rephrase_proba], ["rephrases", args.rephrase_proba]]
        )

    # build up directory for saving output
    os.makedirs(args.output_dir, exist_ok=True)

    # generate a unique identifier for the training model
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
    full_tree_voc = (full_tree, tree_i2w)

    logging.info("====== Setting up NLU Model ======")
    _, encoder_decoder, tokenizer = build_model(args, tree_i2w)

    logging.info("====== Loading Training Dataset ======")
    train_dataset = CAIPDataset(
        tokenizer,
        args,
        prefix="train",
        sampling=True,
        word_noise=args.word_dropout,
        full_tree_voc=full_tree_voc,
    )

    logging.info("====== Loading Validation Datasets ======")
    val_datasets = {}
    for dtype, _ in args.dtype_samples.items():
        val_dataset = CAIPDataset(
            tokenizer,
            args,
            prefix="valid",
            dtype=dtype,
            full_tree_voc=full_tree_voc,
        )
        val_datasets[dtype] = val_dataset

    print(val_datasets)

    logging.info("====== Initializing NLU Model Trainer ======")
    if args.cuda:
        encoder_decoder = encoder_decoder.cuda()
    model_trainer = NLUModelTrainer(
        args, encoder_decoder, tokenizer, model_identifier, full_tree_voc
    )

    logging.info("====== Starting Training Process ======")
    train_steps = 0
    train_loss = 0.0
    train_acc = 0.0
    for epoch in range(args.num_epochs):
        logging.info("Epoch: {}".format(epoch))
        ep_loss, ep_acc, ep_steps, train_dataset = model_trainer.train(epoch, train_dataset)

        train_steps += ep_steps
        train_loss += ep_loss
        train_acc += ep_acc

        # save the model after each epoch
        model_trainer.save_model(epoch, train_dataset)

        logging.info("evaluating model")
        for dtype, val_dataset in val_datasets.items():
            val_loss, val_acc = model_trainer.validate(epoch, val_dataset, dtype)

    print(
        "Training done! Loss: {:.4f} \t Accuracy: {:.4f}".format(
            train_loss / train_steps, train_acc / train_steps
        )
    )
