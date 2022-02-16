"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import argparse
import sys
from data_loaders import SemSegData
import torch
import torch.nn as nn
import torch.optim as optim
import semseg_models as models


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


def semseg_output(S, n, data):
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


def validate(model, DL, loss, optimizer, args):
    model.train()
    losses = []
    correct_num = 0
    total_num = 0
    non_zero_correct = 0
    non_zero_total = 0
    model.eval()
    for b in DL:
        x = b[0]
        y = b[1]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        yhat = model(x)
        ##### calculate acc
        non_zero_idx = y != 0
        non_zero_total += torch.sum(non_zero_idx)

        pred = torch.argmax(yhat, dim=1)
        correct_num += torch.sum(pred == y)
        non_zero_correct += torch.sum((pred == y) * non_zero_idx)
        total_num += torch.numel(y)
        #####
        # loss is expected to not reduce
        preloss = loss(yhat, y)
        mask = torch.zeros_like(y).float()
        u = x.float() + x.float().uniform_(0, 1)
        idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
        mask.view(-1)[idx] = 1
        M = float(idx.size(0))
        # FIXME: eventually need to intersect with "none" tags; want to push loss on labeled empty voxels
        preloss *= mask
        l = preloss.sum() / M
        losses.append(l.item())
    print(
        f"[Valid] Accuracy: {correct_num / total_num}, non empty acc: {non_zero_correct / non_zero_total}"
    )
    return losses


def train_epoch(model, DL, loss, optimizer, args):
    model.train()
    losses = []
    correct_num = 0
    total_num = 0
    non_zero_correct = 0
    non_zero_total = 0
    for b in DL:
        x = b[0]
        y = b[1]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        model.train()
        yhat = model(x)
        ##### calculate acc
        non_zero_idx = y != 0
        non_zero_total += torch.sum(non_zero_idx)

        pred = torch.argmax(yhat, dim=1)
        correct_num += torch.sum(pred == y)
        non_zero_correct += torch.sum((pred == y) * non_zero_idx)
        total_num += torch.numel(y)
        #####
        # loss is expected to not reduce
        preloss = loss(yhat, y)
        mask = torch.zeros_like(y).float()
        u = x.float() + x.float().uniform_(0, 1)
        idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
        mask.view(-1)[idx] = 1
        M = float(idx.size(0))
        # FIXME: eventually need to intersect with "none" tags; want to push loss on labeled empty voxels
        preloss *= mask
        l = preloss.sum() / M
        losses.append(l.item())
        l.backward()
        optimizer.step()
    print(
        f"[Train] Accuracy: {correct_num / total_num}, non empty acc: {non_zero_correct / non_zero_total}"
    )
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", type=int, default=-1, help="no shuffle, keep only debug num examples"
    )
    parser.add_argument("--num_labels", type=int, default=50, help="How many top labels to use")
    parser.add_argument("--num_epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--augment", default="none", help="none or maxshift:K_underdirt:J")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--eval", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--save_model", default="", help="where to save model (nowhere if blank)")
    parser.add_argument(
        "--load_model", default="", help="from where to load model (nowhere if blank)"
    )
    parser.add_argument("--save_logs", default="/dev/null", help="where to save logs")
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=4, help="size of blockid embedding")
    parser.add_argument("--lr", type=float, default=0.02, help="step size for net")
    parser.add_argument(
        "--sample_empty_prob",
        type=float,
        default=0.0001,
        help="prob of taking gradients on empty locations",
    )
    parser.add_argument("--ndonkeys", type=int, default=4, help="workers in dataloader")
    args = parser.parse_args()

    print("loading train data")
    aug = {}
    if args.augment != "none":
        a = args.augment.split("_")
        aug = {t.split(":")[0]: int(t.split(":")[1]) for t in a}
        aug["flip_rotate"] = True
    if args.debug > 0 and len(aug) > 0:
        print("warning debug and augmentation together?")

    train_data = SemSegData(args.data_dir + "training_data.pkl", nexamples=args.debug, augment=aug)
    train_classes = train_data.get_classes()
    valid_data = SemSegData(
        args.data_dir + "validation_data.pkl",
        nexamples=args.debug,
        augment=aug,
        classes=train_classes,
    )

    shuffle = True
    if args.debug > 0:
        shuffle = False

    print("making training dataloader")
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=args.ndonkeys,
    )

    print("making validation dataloader")
    vDL = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batchsize,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=args.ndonkeys,
    )

    args.num_classes = len(train_data.classes["idx2name"])
    args.num_words = 256
    print("making model")
    args.load = False
    if args.load_model != "":
        args.load = True
    model = models.SemSegNet(args, classes=train_data.classes)
    nll = nn.NLLLoss(reduction="none")

    if args.cuda:
        model.cuda()
        nll.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    if args.eval:
        validate(model, rDL, nll, optimizer, args)
        print("[Valid] loss: {}\n".format(sum(losses) / len(losses)))
        exit()

    print("training")
    for m in range(args.num_epochs):
        print(f"========== Epoch {m} =============")
        losses = train_epoch(model, rDL, nll, optimizer, args)
        print("[Train] loss: {}\n".format(sum(losses) / len(losses)))
        losses = validate(model, vDL, nll, optimizer, args)
        print("[Valid] loss: {}\n".format(sum(losses) / len(losses)))
        if args.save_model != "":
            model.save(args.save_model)
