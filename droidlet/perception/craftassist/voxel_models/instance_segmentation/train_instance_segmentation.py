"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import argparse
import sys
from data_loaders import InstSegData
import torch
import torch.nn as nn
import torch.optim as optim
import instseg_models as models


##################################################
# for debugging
##################################################


def print_slices(model, H, r, c, n, data):
    x, y = data[n]
    x = x.unsqueeze(0).cuda()
    yhat = model(x).squeeze()
    print(x[0, c - r : c + r, H, c - r : c + r].cpu())
    print(y[c - r : c + r, H, c - r : c + r])
    _, mm = yhat.max(0)
    print(mm[c - r : c + r, H, c - r : c + r].cpu())


def blocks_from_data(data, n):
    x, y = data[n]
    ids = x.nonzero()
    idl = ids.tolist()
    blocks = [((b[0], b[1], b[2]), (x[b[0], b[1], b[2]].item() + 1, 0)) for b in idl]
    return x, y, blocks


def watcher_output(S, n, data):
    x, y, blocks = blocks_from_data(data, n)
    class_stats = {}
    for i in range(29):
        class_stats[train_data.classes["idx2name"][i]] = len((y == i).nonzero())
        # print(train_data.classes['idx2name'][i], len((y==i).nonzero()))
    a = S._watch_single_object(blocks)
    return class_stats, a


##################################################
# training loop
##################################################


def validate(model, validation_data):
    pass


def train_epoch(model, DL, loss, optimizer, args):
    model.train()
    losses = []
    for b in DL:
        x = b[0]
        s = b[1].unsqueeze(1).float()
        y = b[2].float()
        masks = b[3].float()
        if args.cuda:
            x = x.cuda()
            s = s.cuda()
            y = y.cuda()
            masks = masks.cuda()
        model.train()
        yhat = model(x, s)
        # loss is expected to not reduce
        preloss = loss(yhat, y)
        u = torch.zeros_like(masks).uniform_(0, 1)
        idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
        masks.view(-1)[idx] = 1
        preloss *= masks
        l = preloss.sum() / masks.sum()
        losses.append(l.item())
        l.backward()
        optimizer.step()
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", type=int, default=-1, help="no shuffle, keep only debug num examples"
    )
    parser.add_argument("--num_labels", type=int, default=50, help="How many top labels to use")
    parser.add_argument("--num_epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--num_scales", type=int, default=3, help="if 0 use flat ")
    parser.add_argument("--augment", default="none", help="none or maxshift:K_underdirt:J")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--data_dir", default="/checkpoint/aszlam/minecraft/segmentation_data/")
    parser.add_argument("--save_model", default="", help="where to save model (nowhere if blank)")
    parser.add_argument(
        "--load_model", default="", help="from where to load model (nowhere if blank)"
    )
    parser.add_argument("--save_logs", default="/dev/null", help="where to save logs")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=4, help="size of blockid embedding")
    parser.add_argument("--lr", type=float, default=0.01, help="step size for net")
    parser.add_argument(
        "--sample_empty_prob",
        type=float,
        default=0.01,
        help="prob of taking gradients on empty locations",
    )
    parser.add_argument("--mom", type=float, default=0.0, help="momentum")
    parser.add_argument("--ndonkeys", type=int, default=4, help="workers in dataloader")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.realpath(__file__))

    print("loading train data")
    aug = {}
    if args.augment != "none":
        a = args.augment.split("_")
        aug = {t.split(":")[0]: int(t.split(":")[1]) for t in a}
        aug["flip_rotate"] = True
    if args.debug > 0 and len(aug) > 0:
        print("warning debug and augmentation together?")
    train_data = InstSegData(
        args.data_dir + "training_data.pkl", nexamples=args.debug, augment=aug
    )

    shuffle = True
    if args.debug > 0:
        shuffle = False

    print("making dataloader")
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=args.ndonkeys,
    )

    print("making model")
    args.load = False
    if args.load_model != "":
        args.load = True
    if args.num_scales == 0:
        model = models.FlatInstSegNet(args)
    else:
        model = models.MsInstSegNet(args)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    if args.cuda:
        model.cuda()
        bce.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    #    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("training")
    for m in range(args.num_epochs):
        losses = train_epoch(model, rDL, bce, optimizer, args)
        print(" \nEpoch {} loss: {}".format(m, sum(losses) / len(losses)))
        if args.save_model != "":
            model.save(args.save_model)
