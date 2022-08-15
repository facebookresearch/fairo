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
import vision as models
import time

from torch.utils.tensorboard import SummaryWriter


train_tot_acc_l = []
train_non_air_acc_l = []
train_target_acc_l = []
train_cls_acc_l = {}
valid_tot_acc_l = []
valid_non_air_acc_l = []
valid_target_acc_l = []
valid_cls_acc_l = {}

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


def get_stats(
    stat, text, target_voxel_correct, target_voxel_total, non_air_correct, non_air_total
):
    if text not in stat:
        stat[text] = [(0, 0), (0, 0)]  # [target_voxel, non_air_voxel]
    (prev_correct_target_voxel, prev_total_target_voxel), (
        prev_correct_nonair_voxel,
        prev_total_nonair_voxel,
    ) = stat[text]
    stat[text] = [
        (
            prev_correct_target_voxel + target_voxel_correct,
            prev_total_target_voxel + target_voxel_total,
        ),
        (prev_correct_nonair_voxel + non_air_correct, prev_total_nonair_voxel + non_air_total),
    ]


def stringify_opts(opts):
    data_name = opts.data_dir.strip("/")
    data_name = data_name[data_name.rfind("/") :].strip("/")
    batchsize = opts.batchsize
    lr = opts.lr
    sample_empty_prob = opts.sample_empty_prob
    hidden_dim = opts.hidden_dim
    no_target_prob = opts.no_target_prob
    prob_threshold = opts.prob_threshold
    run_name = opts.run_name
    query_embed = opts.query_embed
    return f"data_{data_name}_batchSize_{batchsize}_lr_{lr}_sampleEmptyProb_{sample_empty_prob}_hiddenDim_{hidden_dim}_noTargetProb_{no_target_prob}_probThreshold_{prob_threshold}_runName_{run_name}_queryembed_{query_embed}"


def validate(model, DL, loss, optimizer, args):
    prob_threshold = args.prob_threshold
    model.eval()
    losses = []
    correct_num = 0
    total_num = 0
    non_zero_correct = 0
    non_zero_total = 0
    model.eval()
    stat = {}
    for b in DL:
        x = b[0]
        y = b[1]
        c = b[2]
        t = b[3]
        text = b[4]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
            t = t.cuda()
        yhat = model(x, t)

        ###### per data wise
        for idx in range(c.size(0)):
            non_zero_idx_i = c[idx] != 0
            non_zero_total_i = torch.sum(non_zero_idx_i)

            pred_i = yhat[idx] > prob_threshold
            non_zero_correct_i = torch.sum((pred_i == y[idx]) * non_zero_idx_i)
            get_stats(stat, text[idx], non_zero_total_i, non_zero_correct_i)
        ######

        ##### calculate acc
        non_zero_idx = c != 0
        non_zero_total += torch.sum(non_zero_idx)
        # print(f"yhat: all: {yhat[0].numel()}, nonzero: {torch.count_nonzero(yhat[0])},  {yhat[0]}")
        pred = yhat > prob_threshold
        correct_num += torch.sum(pred == y)
        non_zero_correct += torch.sum((pred == y) * non_zero_idx)
        total_num += torch.numel(y)
        #####
        # loss is expected to not reduce
        preloss = loss(yhat, y)
        mask = torch.zeros_like(y).float()
        y_clone = y.clone().detach()
        u = y_clone.float() + y_clone.float().uniform_(0, 1)
        idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
        mask.view(-1)[idx] = 1
        M = float(idx.size(0))
        # FIXME: eventually need to intersect with "none" tags; want to push loss on labeled empty voxels
        preloss *= mask
        l = preloss.sum() / M
        losses.append(l.item())
    print(
        f"[Valid] Accuracy: {correct_num / total_num}[{correct_num}/{total_num}], non empty acc: {non_zero_correct / non_zero_total}[{non_zero_correct}/{non_zero_total}]"
    )
    for k, v in stat.items():
        print(f"For shape: {k}, accuracy is: {v[0] / v[1]}[{v[0]}/{v[1]}]")
    return losses


def train_epoch(model, DL, loss, optimizer, args, epoch, validation, summary_writer):
    prob_threshold = args.prob_threshold

    if validation:
        model.eval()
    else:
        model.train()
    losses = []
    correct_num = 0
    total_num = 0
    non_zero_correct = 0
    non_zero_total = 0

    gt_true_correct = 0
    gt_true_total = 0
    stat = {}
    for b in DL:
        x = b[0]
        y = b[1]
        c = b[2]
        t = b[3]
        text = b[4]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            c = c.cuda()
            t = t.cuda()
        model.train()
        yhat = model(x, t)

        ###### per data wise
        for idx in range(y.size(0)):

            # Target voxel
            target_voxel_idx_i = y[idx] != 0
            target_voxel_total_i = torch.sum(target_voxel_idx_i)

            tot_ele_in_data_point = torch.numel(y[idx])
            # print(yhat[idx])
            # print(y[idx])
            pred_i = yhat[idx] > prob_threshold
            target_voxel_correct_i = torch.sum((pred_i == y[idx]) * target_voxel_idx_i)

            ### helper output
            tot_pred_true = torch.sum(pred_i == True)
            tot_pred_correct_all = torch.sum(pred_i == y[idx])

            # Non air voxel
            non_air_idx_i = c[idx] != 0
            non_air_total_i = torch.sum(non_air_idx_i)
            non_air_correct_i = torch.sum((pred_i == y[idx]) * non_air_idx_i)

            # mask = torch.zeros_like(y[idx]).float()
            # y_clone = y[idx].clone().detach()
            # z_clone = y_clone.clone()
            # u = z_clone.float() + y_clone.float().uniform_(0, 1)
            # u2 = z_clone + y_clone
            # idxs = u.gt((1 - args.sample_empty_prob))

            # print(f"DEBUG:, should be all true: {torch.sum(u == u2)}")
            # print(z_clone)
            # print(y_clone)
            # print(idxs)
            # print(f"~~~~~~~~idx: {idx}, tot ele: {tot_ele_in_data_point}, tot should be true: {non_zero_total_i}, tot pred true: {tot_pred_true}, tot pred true correctly with air: {tot_pred_correct_all}, tot pred true correctly without air: {non_zero_correct_i}~~~~~~~~~~")
            # print(pred_i)
            # print(y[idx])
            ### helper output
            get_stats(
                stat,
                text[idx],
                target_voxel_correct_i.item(),
                target_voxel_total_i.item(),
                non_air_correct_i.item(),
                non_air_total_i.item(),
            )
        ######

        ##### calculate acc
        non_zero_idx = c != 0
        non_zero_total += torch.sum(non_zero_idx)

        pred = yhat > prob_threshold
        correct_num += torch.sum(pred == y)
        non_zero_correct += torch.sum((pred == y) * non_zero_idx)
        total_num += torch.numel(y)

        gt_true_idx = y != 0
        gt_true_total += torch.sum(gt_true_idx)
        gt_true_correct += torch.sum((pred == y) * gt_true_idx)
        #####

        # loss is expected to not reduce
        preloss = loss(yhat, y)
        mask = torch.zeros_like(y).float()

        ##### HERE CHANGE Y TO C, CHANGE IT BACK!!!!!!!!!!!!
        y_clone = c.clone().detach()
        z_clone = y_clone.clone()
        u = y_clone.float() + z_clone.float().uniform_(0, 1)
        idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
        mask.view(-1)[idx] = 1
        M = float(idx.size(0))
        # FIXME: eventually need to intersect with "none" tags; want to push loss on labeled empty voxels
        preloss *= mask
        l = preloss.sum() / M
        losses.append(l.item())
        if not validation:
            l.backward()
            optimizer.step()
            # print(f"non zero total: {non_zero_total}, total: {total_num}")
    if not validation:
        print(
            f"[Train] Accuracy: {correct_num / total_num}[{correct_num}/{total_num}], non air acc: {non_zero_correct / non_zero_total}[{non_zero_correct}/{non_zero_total}], gt should be true acc: {gt_true_correct / gt_true_total}[{gt_true_correct}/{gt_true_total}]"
        )
        train_tot_acc_l.append((correct_num / total_num).item())

        train_non_air_acc_l.append((non_zero_correct / non_zero_total).item())

        train_target_acc_l.append((gt_true_correct / gt_true_total).item())

        for k, v in stat.items():
            if k not in train_cls_acc_l.keys():
                train_cls_acc_l[k] = []
            # train_cls_acc_l[k].append((v[0] / v[1]).item())
            print(
                f"For shape: {k}, target voxel acc: {(v[0][0] / (v[0][1] + 1)):.2f}[{v[0][0]}/{v[0][1]+1}], non air voxel acc: {(v[1][0] / (v[1][1] + 1)):.2f}[{v[1][0]}/{v[1][1]+1}]"
            )

        precision_all = correct_num / total_num
        precision_occupied = non_zero_correct / non_zero_total
        recall = gt_true_correct / gt_true_total
        f1_all = 2 * precision_all * recall / (precision_all + recall)
        f1_occupied = 2 * precision_occupied * recall / (precision_occupied + recall)
        summary_writer.add_scalar("Precision_All[Accuracy]/Train", precision_all, epoch)
        summary_writer.add_scalar("Precision_Occupied/Train", precision_occupied, epoch)
        summary_writer.add_scalar("Recall/Train", recall, epoch)
        summary_writer.add_scalar("F1_All/Train", f1_all, epoch)
        summary_writer.add_scalar("F1_Occupied/Train", f1_occupied, epoch)
        summary_writer.add_scalar("Loss/Train", sum(losses) / len(losses), epoch)
        # print(f"train_tot_acc_l\n{train_tot_acc_l}")
        # print(f"train_non_air_acc_l\n{train_non_air_acc_l}")
        # print(f"train_target_acc_l\n{train_target_acc_l}")
        # print(f"train_cls_acc_l\n{train_cls_acc_l}")
    else:
        print(
            f"[Valid] Accuracy: {correct_num / total_num}[{correct_num}/{total_num}], non air acc: {non_zero_correct / non_zero_total}[{non_zero_correct}/{non_zero_total}], gt should be true acc: {gt_true_correct / gt_true_total}[{gt_true_correct}/{gt_true_total}]"
        )
        valid_tot_acc_l.append((correct_num / total_num).item())
        valid_non_air_acc_l.append((non_zero_correct / non_zero_total).item())
        valid_target_acc_l.append((gt_true_correct / gt_true_total).item())
        for k, v in stat.items():
            if k not in valid_cls_acc_l.keys():
                valid_cls_acc_l[k] = []
            # valid_cls_acc_l[k].append((v[0] / v[1]).item())
            print(
                f"For shape: {k}, target voxel acc: {(v[0][0] / (v[0][1] + 1)):.2f}[{v[0][0]}/{v[0][1]+1}], non air voxel acc: {(v[1][0] / (v[1][1] + 1)):.2f}[{v[1][0]}/{v[1][1]+1}]"
            )

        precision_all = correct_num / total_num
        precision_occupied = non_zero_correct / non_zero_total
        recall = gt_true_correct / gt_true_total
        f1_all = 2 * precision_all * recall / (precision_all + recall)
        f1_occupied = 2 * precision_occupied * recall / (precision_occupied + recall)
        summary_writer.add_scalar("Precision_All[Accuracy]/Valid", precision_all, epoch)
        summary_writer.add_scalar("Precision_Occupied/Valid", precision_occupied, epoch)
        summary_writer.add_scalar("Recall/Valid", recall, epoch)
        summary_writer.add_scalar("F1_All/Valid", f1_all, epoch)
        summary_writer.add_scalar("F1_Occupied/Valid", f1_occupied, epoch)
        summary_writer.add_scalar("Loss/Valid", sum(losses) / len(losses), epoch)
        # print(f"valid_tot_acc_l\n{valid_tot_acc_l}")
        # print(f"valid_non_air_acc_l\n{valid_non_air_acc_l}")
        # print(f"valid_target_acc_l\n{valid_target_acc_l}")
        # print(f"valid_cls_acc_l\n{valid_cls_acc_l}")

    return losses


def get_parser():
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
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="prob threshold of being considered positive",
    )
    parser.add_argument(
        "--no_target_prob",
        type=float,
        default=0.2,
        help="prob of no target object in the scene",
    )
    parser.add_argument("--ndonkeys", type=int, default=4, help="workers in dataloader")

    parser.add_argument("--run_name", default="", help="unique name to identify this run")
    parser.add_argument(
        "--visualization_dir",
        default="/checkpoint/yuxuans/vis_dir",
        help="path to store tensorboard graphs",
    )
    parser.add_argument("--query_embed", default="lut", help="lut or bert")
    return parser


def main(args):

    # torch.cuda.set_device(1)
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # args = parser.parse_args()

    print("loading train data")
    aug = {}
    if args.augment != "none":
        a = args.augment.split("_")
        aug = {t.split(":")[0]: int(t.split(":")[1]) for t in a}
        aug["flip_rotate"] = True
    if args.debug > 0 and len(aug) > 0:
        print("warning debug and augmentation together?")

    train_data = SemSegData(
        args.data_dir + "training_data.pkl",
        nexamples=args.debug,
        augment=aug,
        no_target_prob=args.no_target_prob,
        query_embed=args.query_embed,
    )
    train_classes = train_data.get_classes()
    valid_data = SemSegData(
        args.data_dir + "validation_data.pkl",
        nexamples=args.debug,
        augment=aug,
        classes=train_classes,
        no_target_prob=args.no_target_prob,
        query_embed=args.query_embed,
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
    # nll = nn.NLLLoss(reduction="none")
    bceloss = nn.BCELoss(reduction="none")

    if args.cuda:
        model.cuda()
        bceloss.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    if args.eval:
        validate(model, rDL, bceloss, optimizer, args)
        print("[Valid] loss: {}\n".format(sum(losses) / len(losses)))
        exit()

    unique_name = stringify_opts(args)
    writer = SummaryWriter(f"{args.visualization_dir}/{unique_name}")

    train_loss_l = []
    valid_loss_l = []
    print("training")
    for epoch in range(args.num_epochs):
        print(f"========== Epoch {epoch} =============")
        losses = train_epoch(
            model, rDL, bceloss, optimizer, args, epoch, validation=False, summary_writer=writer
        )
        print("[Train] loss: {}\n".format(sum(losses) / len(losses)))
        train_loss_l.append((sum(losses) / len(losses)))
        # print(f"[Train] loss list: \n {train_loss_l}")
        losses = train_epoch(
            model, vDL, bceloss, optimizer, args, epoch, validation=True, summary_writer=writer
        )
        print("[Valid] loss: {}\n".format(sum(losses) / len(losses)))
        valid_loss_l.append((sum(losses) / len(losses)))
        # print(f"[Valid] loss list: \n {valid_loss_l}")
        if args.save_model != "":
            model.save(args.save_model)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn", force=True)
    parser = get_parser()
    args = parser.parse_args()
    main(args)
