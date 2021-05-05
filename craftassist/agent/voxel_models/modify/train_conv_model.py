"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os

print(os.getcwd())
import sys

sys.path = [""] + sys.path
from shape_transform_dataloader import ModifyData
import torch

# import torch.nn as nn
import torch.optim as optim
import conv_models as models

# predict allowed blocks
# quantize to nearest in allowed set


def format_stats(stats_dict):
    status = "STATS :: epoch@{} | loss@{}".format(stats_dict["epoch"], stats_dict["loss"])
    return status


# FIXME allow restarting optimizer via opts
def get_optimizer(args, model, lr=None, allow_load=True):
    if not lr:
        lr = args.lr
    sd = None
    if allow_load and args.load_model_dir != "" and model.loaded_from is not None:
        fname = os.path.basename(model.loaded_from)
        fdir = os.path.dirname(model.loaded_from)
        optimizer_path = os.path.join(fdir, "optim." + fname)
        try:
            sd = torch.load(optimizer_path)
        except:
            print("warning, unable to load optimizer from ")
            print(optimizer_path)
            print("restarting optimzier")
    if args.optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optim_type == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    if sd:
        try:
            optimizer.load_state_dict(sd)
        except:
            print("warning, optimizer from ")
            print(optimizer_path)
            print("appears corrupted, restarting optimzier")

    return optimizer


def lower_lr(model, args, optimizer, new_lr):
    new_optimizer = get_optimizer(args, model, lr=new_lr, allow_load=False)
    sd = optimizer.state_dict()
    new_optimizer.load_state_dict(sd)
    return new_optimizer


def save_optimizer(optimizer, fpath):
    fname = os.path.basename(fpath)
    fdir = os.path.dirname(fpath)
    optimizer_path = os.path.join(fdir, "optim." + fname)
    torch.save(optimizer.state_dict(), optimizer_path)


##################################################
# for debugging
##################################################


def draw_all(n):
    sp.drawPlotly(unhash_volume(b[1][n], 20))  # noqa
    draw_color_hash(maxi[n], vis)  # noqa
    sp.drawPlotly(unhash_volume(b[2][n], 20), title=args._train_data.print_text(b[0][n]))  # noqa


def unhash_volume(x, max_meta):
    meta = x % max_meta
    bid = (x - meta) // max_meta
    return torch.stack((bid, meta), 3).cpu().numpy()


def get_im(z, embedding, allowed_idxs):
    weight = embedding.weight.index_select(0, allowed_idxs)
    k = weight.shape[0]
    d = weight.shape[1]
    scores = torch.nn.functional.conv3d(z, weight.view(k, d, 1, 1, 1))
    scores = scores.permute(0, 2, 3, 4, 1).contiguous()
    maxs, maxi = scores.max(4)
    return allowed_idxs[maxi]


def f_get_im_and_draw(w, z, embedding, allowed_idxs, i, train_data):
    B = get_im(z, embedding, allowed_idxs)
    text = "  ".join(
        [
            train_data.dictionary["i2w"][l.item()]
            for l in w[i]
            if l.item() < len(train_data.dictionary["i2w"])
        ]
    )
    idm = unhash_volume(B[i], train_data.max_meta)
    sp.drawPlotly(idm, title=" ".join(text))


##################################################
# training loop
##################################################


def find_lvars(model, x, y, words, loss_fn, args):
    with torch.no_grad():
        losses = torch.zeros(x.shape[0], args.num_lvars, device=x.device)
        for i in range(args.num_lvars):
            lvars = torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.long).fill_(i)
            z = model(x, words, lvars)
            if args.color_io >= 0 and args.color_hash < 0:
                l = torch.nn.functional.mse_loss(z, model.input_embed)
            elif args.color_hash > 0:
                l = loss_fn(y, z, args.color_hash)
            losses[:, i] = l
        minval, mini = losses.min(1)
        return mini


def validate(model, validation_data):
    pass


def train_epoch(model, DL, loss_fn, optimizer, args):
    model.train()
    losses = []
    for b in DL:
        optimizer.zero_grad()
        words = b[0]
        x = b[1]
        y = b[2]
        if args.cuda:
            words = words.cuda()
            x = x.cuda()
            y = y.cuda()
        model.train()
        allowed_idxs = torch.unique(y)
        if args.ae:
            z = model(y)
        else:
            lvars = find_lvars(model, x, y, words, loss_fn, args)
            z = model(x, words, lvars)
        if args.color_io >= 0 and args.color_hash < 0:
            l = torch.nn.functional.mse_loss(z, model.input_embed)
        elif args.color_hash > 0:
            l = loss_fn(y, z, args.color_hash)
        else:
            l = loss_fn(y, z, allowed_idxs)
        losses.append(l.item())
        l.backward()
        optimizer.step()
    return losses


def main(args):
    print("loading train data")

    ################# FIXME!!!!
    args.allow_same = False
    args.nexamples = 1024
    args.tform_weights = {
        "thicker": 1.0,
        "scale": 1.0,
        "rotate": 1.0,
        "replace_by_block": 1.0,
        #                          'replace_by_n': 1.0,
        "replace_by_halfspace": 1.0,
        "fill": 1.0,
    }

    train_data = ModifyData(args, dictionary=args.load_dictionary)

    shuffle = True
    if args.debug > 0:
        num_workers = 0
        shuffle = False
    else:
        num_workers = args.ndonkeys

    print("making dataloader")
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batchsize,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )

    print("making models")
    args.num_words = train_data.padword + 1
    args.word_padding_idx = train_data.padword

    ###########################################
    # args.pool = 8 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    args.pool = None
    ###########################################

    if args.ae:
        model = models.AE(args)
    else:
        model = models.SimpleConv(args, pool=args.pool)
    if args.cuda:
        model.cuda()
    else:
        print("warning:  no cuda")

    ############
    if args.color_hash > 0:
        loss_fn = models.ConvNLL(max_meta=args.max_meta)
        if args.cuda:
            loss_fn.cuda()
    else:
        loss_fn = models.ConvDistributionMatch(model.embedding, subsample_zeros=0.01)
    ############

    optimizer = get_optimizer(args, model)

    args._model = model
    args._train_data = train_data
    print("training")
    minloss = 1000000
    badcount = 0
    lr = args.lr
    for m in range(args.num_epochs):
        losses = train_epoch(model, rDL, loss_fn, optimizer, args)
        mean_epoch_loss = sum(losses) / len(losses)
        status = format_stats({"epoch": m, "loss": mean_epoch_loss})
        print(status)
        if args.save_model_dir != "":
            fpath = models.model_filename_from_opts(
                args, savedir=args.save_model_dir, uid=args.save_model_uid
            )
            model.save(fpath)
            save_optimizer(optimizer, fpath)
        if mean_epoch_loss < 0.99 * minloss:
            minloss = mean_epoch_loss
            badcount = 0
        else:
            badcount = badcount + 1
        if badcount > args.lr_patience and args.lr_decay < 1.0:
            lr = args.lr_decay * lr
            optimizer = lower_lr(model, args, optimizer, lr)
            print("lowered lr to " + str(lr))
            badcount = 0

    return model, train_data, rDL


if __name__ == "__main__":
    import argparse
    from voxel_models.plot_voxels import SchematicPlotter, draw_rgb, draw_color_hash  # noqa
    import visdom

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", type=int, default=-1, help="no shuffle, keep only debug num examples"
    )
    parser.add_argument("--num_examples", type=int, default=1024, help="num examples in an epoch")
    parser.add_argument("--num_epochs", type=int, default=1500, help="training epochs")
    parser.add_argument("--last_layer_sigmoid", type=int, default=1, help="do sigmoid if 1")
    parser.add_argument(
        "--color_hash", type=int, default=-1, help="if > 0 hash color cube into bins"
    )
    parser.add_argument("--num_lvars", type=int, default=10, help="number of random vars")
    parser.add_argument(
        "--lr_patience", type=int, default=8, help="how many epochs to wait before decreasing lr"
    )
    parser.add_argument("--lr_decay", type=float, default=1.0, help="lr decrease multiple")
    parser.add_argument("--num_layers", type=int, default=4, help="num conv layers")
    # parser.add_argument("--augment", default="none", help="none or maxshift:K_underdirt:J")
    parser.add_argument("--sidelength", type=int, default=32, help="sidelength for dataloader")
    parser.add_argument(
        "--color_io",
        type=int,
        default=2,
        help="if 2 input uses color-alpha, 1 intensity, 0 bw, -1 emebdding",
    )
    parser.add_argument("--ae", action="store_true", help="plain ae")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument(
        "--max_meta", type=int, default=20, help="allow that many meta values when hashing idmeta"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--data_dir", default="")
    parser.add_argument(
        "--save_model_dir", default="", help="where to save model (nowhere if blank)"
    )
    parser.add_argument(
        "--load_model_dir", default="", help="from where to load model (nowhere if blank)"
    )
    parser.add_argument(
        "--load_strict",
        action="store_true",
        help="error if model to load doesn't exist.  if false just builds new",
    )
    parser.add_argument(
        "--save_model_uid", default="", help="unique identifier on top of options-name"
    )
    parser.add_argument(
        "--words_length", type=int, default=12, help="sentence pad length.  FIXME?"
    )
    parser.add_argument("--optim_type", default="adam", help="optim type, adam, adagrad, sgd")
    parser.add_argument("--save_logs", default="/dev/null", help="where to save logs")
    parser.add_argument(
        "--residual_connection", type=int, default=0, help="if bigger than 0 use resnet-style"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="size of hidden dim")
    parser.add_argument("--embedding_dim", type=int, default=16, help="size of blockid embedding")
    parser.add_argument(
        "--load_dictionary",
        default="/private/home/aszlam/junk/word_modify_word_ids.pk",
        help="where to get word dict",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="step size for net")
    parser.add_argument(
        "--sbatch", action="store_true", help="cluster run mode, no visdom, formatted stdout"
    )
    parser.add_argument(
        "--sample_empty_prob",
        type=float,
        default=0.01,
        help="prob of taking gradients on empty locations",
    )
    parser.add_argument("--ndonkeys", type=int, default=8, help="workers in dataloader")
    args = parser.parse_args()
    if not args.sbatch:
        vis = visdom.Visdom(server="http://localhost")
        sp = SchematicPlotter(vis)

    main(args)
