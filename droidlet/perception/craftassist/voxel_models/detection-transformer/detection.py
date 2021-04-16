"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import argparse
import builtins
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import datasets
import to_coco_api
import utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--eval_skip", default=1, type=int, help='do evaluation every "eval_skip" frames'
    )
    parser.add_argument("--schedule", default="step", type=str, choices=("step", "multistep"))

    # model params
    parser.add_argument("--model_file", default="model_parallel")
    parser.add_argument(
        "--mask_model", default="none", type=str, choices=("none", "smallconv", "v2")
    )
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=248, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
    )
    parser.add_argument("--set_cost_class", default=1, type=float)
    parser.add_argument("--set_cost_bbox", default=5, type=float)
    parser.add_argument("--set_cost_giou", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=1, type=float)
    parser.add_argument("--backbone", default="semseg", type=str)
    # parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument("--position_embedding", default="v2", type=str, choices=("v1", "v2", "v3"))
    parser.add_argument("--resample_features_to_size", default=-1, type=int)
    parser.add_argument("--eos_coef", default=0.1, type=float)
    parser.add_argument("--num_queries", default=99, type=int)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--pass_pos_and_query", action="store_true")
    parser.add_argument("--dilation", action="store_true")

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument(
        "--crowdfree", action="store_true", help="Remove crowd images from training on COCO"
    )
    parser.add_argument("--masks", action="store_true")

    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.mask_model != "none":
        args.masks = True
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessor = build_model(args)
    postprocessor.rescale_to_orig_size = True  # for evaluation
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = builtins.sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # optimizer = torch.optim.Adam(model.parameters())
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n],
            "lr": args.lr_backbone,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    if args.schedule == "step":
        lr_scheduler = StepLR(optimizer, args.lr_drop)
    elif args.schedule == "multistep":
        milestones = list(range(args.lr_drop, args.epochs, 50))
        lr_scheduler = MultiStepLR(optimizer, gamma=0.5, milestones=milestones)

    dataset_train = build_dataset(image_set="trainval", args=args)
    dataset_val = build_dataset(image_set="test", args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = to_coco_api.get_coco_api_from_dataset(coco_val)
    else:
        base_ds = None  # to_coco_api.get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessor,
            data_loader_val,
            base_ds,
            device,
            eval_bbox=True,
            eval_masks=args.masks,
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        # if epoch % args.eval_skip == 0:
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessor, data_loader_val, base_ds, device, eval_bbox=True, eval_masks=args.masks
        #     )
        # else:
        #     test_stats, coco_evaluator = {}, None
        test_stats, coco_evaluator = {}, None

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name
                        )

            with (output_dir / "log_tb.txt").open("a") as f:
                f.write(f"TORCHBOARD_METRICS[epoch] = {epoch}\n")
                for k, v in vars(args).items():
                    f.write(f"TORCHBOARD_METRICS[{k}] = {v}")
                for key in log_stats:
                    v = log_stats[key]
                    if isinstance(v, list):
                        for i, vi in enumerate(v):
                            f.write(f"TORCHBOARD_METRICS[{key}_{i}] = {vi}\n")
                    else:
                        f.write(f"TORCHBOARD_METRICS[{key}] = {v}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Set transformer detector", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
