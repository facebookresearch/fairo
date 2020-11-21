"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import random
import os
import sys
import argparse
import torch
import string
import json
from shutil import copyfile
from inspect import currentframe, getframeinfo
from datetime import datetime

import models
import combined_dataset as cd

"""
General Training Utils
"""


def pretty_log(log_string):
    cf = currentframe().f_back
    filename = getframeinfo(cf).filename.split("/")[-1]
    print(
        "{} {}:{} {}".format(
            datetime.now().strftime("%m/%d/%Y %H:%M:%S"), filename, cf.f_lineno, log_string
        )
    )
    sys.stdout.flush()


def prepare_variables(b, opts):
    X = b.long()
    if opts["cuda"]:
        X = X.cuda()
    return X


def set_modules(tms, train):
    for m in ["context_net", "seg_net", "score_module", "seg_direction_net"]:
        if m not in tms:
            continue
        if train:
            tms[m].train()
        else:
            tms[m].eval()


def multitensor_collate_fxn(x):
    regroup_tensors = {n: [] for n in x[0].keys()}
    use_names = list(x[0].keys())
    for t_dict in x:
        use_names = [n for n in use_names if n in t_dict]
        for n, t in t_dict.items():
            if n not in regroup_tensors:
                continue
            regroup_tensors[n].append(t.unsqueeze(0))
    use_names = set(use_names)
    batched_tensors = {
        n: torch.cat([t.float() for t in tl])
        for n, tl in regroup_tensors.items()
        if n in use_names
    }
    return batched_tensors


def get_dataloader(dataset, opts, collate_fxn):
    def init_fn(wid):
        np.random.seed(torch.initial_seed() % (2 ** 32))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opts["batchsize"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opts["num_workers"],
        worker_init_fn=init_fn,
        collate_fn=collate_fxn,
    )


def setup_dataset_and_loader(opts):
    extra_params = {
        "drop_perc": opts.get("drop_perc", 0.0),
        "fixed_size": opts.get("fixed_size", None),
        "shape_type": opts.get("shape_type", "random"),
        "max_shift": opts.get("max_shift", 0),
        "ground_type": opts.get("ground_type", None),
        "random_ground_height": opts.get("random_ground_height", False),
    }
    config = opts.get("dataset_config", None)
    if config is not None:
        print(">> Loaded config from", config)
        with open(config, "r") as f:
            config = json.load(f)
    dataset = cd.CombinedData(
        nexamples=opts["epochsize"],
        useid=opts["useid"],
        extra_params=extra_params,
        ratios=parse_dataset_ratios(opts),
        config=config,
    )
    dataloader = get_dataloader(dataset=dataset, opts=opts, collate_fxn=multitensor_collate_fxn)
    return dataset, dataloader


def get_scores_from_datapoint(tms, batch, opts):
    batch = {k: prepare_variables(t, opts) for k, t in batch.items()}
    c_embeds = tms["context_net"](batch)
    s_embeds = tms["seg_net"](batch)
    if "seg_direction_net" in tms:
        if s_embeds.dim() > 2:
            s_embeds = s_embeds.squeeze()
        if s_embeds.dim() == 1:
            s_embeds = s_embeds.unsqueeze(0)
        batch["s_embeds"] = s_embeds
        s_embeds = tms["seg_direction_net"](batch)
    scores = tms["score_module"]({"c_embeds": c_embeds, "s_embeds": s_embeds})
    return scores


def get_scores_and_target_from_datapoint(tms, batch, opts):
    batch = {k: prepare_variables(t, opts) for k, t in batch.items()}
    batch["target"] = batch["target"].squeeze()

    tms["optimizer"].zero_grad()
    c_embeds = tms["context_net"](batch)
    s_embeds = tms["seg_net"](batch)
    if "seg_direction_net" in tms:
        if s_embeds.dim() > 2:
            s_embeds = s_embeds.squeeze()
        if s_embeds.dim() == 1:
            s_embeds = s_embeds.unsqueeze(0)
        batch["s_embeds"] = s_embeds
        s_embeds = tms["seg_direction_net"](batch)
    scores = tms["score_module"]({"c_embeds": c_embeds, "s_embeds": s_embeds})
    return batch["target"], scores


"""
Checkpointing
"""


def check_and_print_opts(curr_opts, old_opts):
    mismatches = []
    print(">> Options:")
    for opt, val in curr_opts.items():
        if opt and val:
            print("   - {:>20}: {:<30}".format(opt, val))
        else:
            print("   - {}: {}".format(opt, val))

        if old_opts and opt in old_opts and old_opts[opt] != val:
            mismatches.append((opt, val, old_opts[opt]))
            print("")

    if len(mismatches) > 0:
        print(">> Mismatching options:")
        for m in mismatches:
            if any([mv is None for mv in m]):
                continue
            print("   - {:>20}: new '{:<10}' != old '{:<10}'".format(m[0], m[1], m[2]))
            print("")
    return True if len(mismatches) > 0 else False


def load_context_segment_checkpoint(
    checkpoint_path, opts, backup=True, verbose=False, use_new_opts=False
):
    if not os.path.isfile(checkpoint_path):
        check_and_print_opts(opts, None)
        return {}

    if backup:
        random_uid = "".join(
            [random.choice(string.ascii_letters + string.digits) for n in range(4)]
        )
        backup_path = checkpoint_path + ".backup_" + random_uid
        copyfile(checkpoint_path, backup_path)
        print(">> Backing up checkpoint before loading and overwriting:")
        print("        {}\n".format(backup_path))

    checkpoint = torch.load(checkpoint_path)

    if verbose:
        print(">> Loading model from checkpoint {}".format(checkpoint_path))
        for opt, val in checkpoint["metadata"].items():
            print("    - {:>20}: {:<30}".format(opt, val))
        print("")
        check_and_print_opts(opts, checkpoint["options"])

    checkpoint_opts_dict = checkpoint["options"]
    if type(checkpoint_opts_dict) is not dict:
        checkpoint_opts_dict = vars(checkpoint_opts_dict)

    if not use_new_opts:
        for opt, val in checkpoint_opts_dict.items():
            opts[opt] = val
    else:
        print(">> Using passed in options")
    print(">> Geoscorer model opts:", opts)

    trainer_modules = models.create_context_segment_modules(opts)
    trainer_modules["opts"] = checkpoint_opts_dict
    trainer_modules["context_net"].load_state_dict(checkpoint["model_state_dicts"]["context_net"])
    trainer_modules["seg_net"].load_state_dict(checkpoint["model_state_dicts"]["seg_net"])
    trainer_modules["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
    if opts.get("seg_direction_net", False):
        trainer_modules["seg_direction_net"].load_state_dict(
            checkpoint["model_state_dicts"]["seg_direction_net"]
        )
    move_to_device(opts, trainer_modules)
    return trainer_modules


def get_context_segment_trainer_modules(
    opts, checkpoint_path=None, backup=False, verbose=False, use_new_opts=False
):
    trainer_modules = load_context_segment_checkpoint(
        checkpoint_path, opts, backup, verbose, use_new_opts=use_new_opts
    )
    print("len trainer_modules", len(trainer_modules))

    if len(trainer_modules) == 0:
        if checkpoint_path is not None:
            print(">> checkpoint path does not exist, initializing new model", checkpoint_path)
        trainer_modules = models.create_context_segment_modules(opts)
        trainer_modules["opts"] = opts
    return trainer_modules


def move_to_device(opts, tms):
    if opts["cuda"]:
        tms["context_net"].cuda()
        tms["seg_net"].cuda()
        if opts.get("seg_direction_net", False):
            tms["seg_direction_net"].cuda()
    else:
        tms["context_net"].cpu()
        tms["seg_net"].cpu()
        if opts.get("seg_direction_net", False):
            tms["seg_direction_net"].cpu()


def save_checkpoint(tms, metadata, opts, path):
    model_dict = {"context_net": tms["context_net"], "seg_net": tms["seg_net"]}
    if opts.get("seg_direction_net", False):
        model_dict["seg_direction_net"] = tms["seg_direction_net"]

    # Add all models to dicts and move state to cpu
    state_dicts = {}
    for model_name, model in model_dict.items():
        state_dicts[model_name] = model.state_dict()
        for n, s in state_dicts[model_name].items():
            state_dicts[model_name][n] = s.cpu()

    # Save to path
    torch.save(
        {
            "metadata": metadata,
            "model_state_dicts": state_dicts,
            "optimizer_state_dict": tms["optimizer"].state_dict(),
            "options": opts,
        },
        path,
    )


"""
Parser Arguments
"""


def get_train_parser():
    parser = argparse.ArgumentParser()

    # Base Training Flags
    parser.add_argument("--cuda", type=int, default=1, help="0 for cpu")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument(
        "--epochsize", type=int, default=1000, help="number of examples in an epoch"
    )
    parser.add_argument("--nepoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="step size for net")
    parser.add_argument(
        "--optim", type=str, default="adagrad", help="optim type to use (adagrad|sgd|adam)"
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum")
    parser.add_argument("--checkpoint", default="", help="where to save model")
    parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument(
        "--backup", action="store_true", help="backup the checkpoint path before saving to it"
    )
    parser.add_argument(
        "--visualize_epochs", action="store_true", help="use visdom to visualize progress"
    )

    # Model Flags
    parser.add_argument("--hidden_dim", type=int, default=64, help="size of hidden dim")
    parser.add_argument("--num_layers", type=int, default=3, help="num layers")
    parser.add_argument(
        "--blockid_embedding_dim", type=int, default=8, help="size of blockid embedding"
    )
    parser.add_argument("--context_sidelength", type=int, default=32, help="size of cube")
    parser.add_argument("--useid", action="store_true", help="use blockid")
    parser.add_argument(
        "--num_words", type=int, default=256, help="number of words for the blockid embeds"
    )

    # Dataset Flags
    parser.add_argument(
        "--dataset_ratios", type=str, default="shape:1.0", help="comma separated name:prob"
    )
    parser.add_argument(
        "--drop_perc", type=float, default=0.5, help="perc segs to drop from inst_seg"
    )
    parser.add_argument(
        "--max_shift", type=int, default=6, help="max amount to offset shape_dir target"
    )
    parser.add_argument(
        "--ground_type", type=str, default=None, help="ground type to include in datasets"
    )
    parser.add_argument(
        "--random_ground_height", action="store_true", help="false means height is max"
    )
    parser.add_argument(
        "--shape_type", type=str, default="random", help="set both shapes to same type"
    )
    parser.add_argument("--fixed_size", type=int, default=None, help="fix shape size")
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="for more complex training"
    )

    # Directional Placement Flags
    parser.add_argument("--spatial_embedding_dim", type=int, default=8, help="size of spatial emb")
    parser.add_argument("--output_embedding_dim", type=int, default=8, help="size of output emb")
    parser.add_argument("--seg_direction_net", action="store_true", help="use segdirnet module")
    parser.add_argument("--seg_use_viewer_pos", action="store_true", help="use viewer pos in seg")
    parser.add_argument("--seg_use_direction", action="store_true", help="use direction in seg")
    parser.add_argument("--num_seg_dir_layers", type=int, default=3, help="num segdir net layers")
    parser.add_argument(
        "--cont_use_direction", action="store_true", help="use direction in context"
    )
    parser.add_argument(
        "--cont_use_xyz_from_viewer_look",
        action="store_true",
        help="use xyz position relative to viewer look in context emb",
    )
    return parser


def parse_dataset_ratios(opts):
    ratios_str = opts["dataset_ratios"]
    ratio = {}
    try:
        l_s = ratios_str.split(",")
        print("\n>> Dataset ratio flag (not used if config is provided):")
        for t in l_s:
            name, prob = t.split(":")
            ratio[name] = float(prob)
            print("  -     {}: {}".format(name, prob))
        print("")
    except:
        raise Exception("Failed to parse the dataset ratio string {}".format(ratios_str))
    return ratio
