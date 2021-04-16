"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
from shape_transform_dataloader import ModifyData
import torch
from torch import distributions

# import torch.nn as nn
import torch.optim as optim
import conv_models as models
from gan_trainer import Trainer


# predict allowed blocks
# quantize to nearest in allowed set


def format_stats(stats_dict):
    status = "STATS :: epoch@{} | gloss@{} |  dloss@{}".format(
        stats_dict["epoch"], stats_dict["gloss"], stats_dict["dloss"]
    )
    return status


# FIXME allow restarting optimizer via opts
def get_optimizer(args, model):
    sds = None
    if args.load_model_dir != "" and model.loaded_from is not None:
        fname = os.path.basename(model.loaded_from)
        fdir = os.path.dirname(model.loaded_from)
        optimizer_path = os.path.join(fdir, "optim." + fname)
        try:
            sds = torch.load(optimizer_path)
            sd_g = sds["g"]
            sd_d = sds["d"]
        except:
            print("warning, unable to load optimizer from ")
            print(optimizer_path)
            print("restarting optimzier")
    if args.optim_type == "adam":
        optimizer_d = optim.Adam(model.D.parameters(), lr=args.lr_d)
        optimizer_g = optim.Adam(model.G.parameters(), lr=args.lr_g)
    elif args.optim_type == "adagrad":
        optimizer_d = optim.Adagrad(model.D.parameters(), lr=args.lr_d)
        optimizer_g = optim.Adagrad(model.G.parameters(), lr=args.lr_g)
    elif args.optim_type == "rmsprop":
        optimizer_d = optim.RMSprop(model.D.parameters(), lr=args.lr_d, alpha=0.99, eps=1e-8)
        optimizer_g = optim.RMSprop(model.G.parameters(), lr=args.lr_g, alpha=0.99, eps=1e-8)
    else:
        optimizer_d = optim.SGD(model.D.parameters(), lr=args.lr_d)
        optimizer_g = optim.SGD(model.G.parameters(), lr=args.lr_g)
    if sds:
        try:
            optimizer_d.load_state_dict(sd_d)
            optimizer_g.load_state_dict(sd_g)
        except:
            print("warning, optimizer from ")
            print(optimizer_path)
            print("appears corrupted, restarting optimzier")

    return optimizer_d, optimizer_g


def save_optimizer(optimizer_d, optimizer_g, fpath):
    fname = os.path.basename(fpath)
    fdir = os.path.dirname(fpath)
    optimizer_path = os.path.join(fdir, "optim." + fname)
    torch.save({"d": optimizer_d.state_dict(), "g": optimizer_g.state_dict()}, optimizer_path)


##################################################
# for debugging
##################################################


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


def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == "uniform":
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == "gauss":
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def validate(model, validation_data):
    pass


def train_epoch(model, DL, trainer, args):
    model.train()
    losses = {"g": [], "d": []}
    for b in DL:
        words = b[0]
        x = b[1]
        y = b[2]
        if args.cuda:
            words = words.cuda()
            x = x.cuda()
            y = y.cuda()
        zdist = get_zdist("gauss", args.zdim, device=x.device)

        x_real = models.fake_embedding_fwd(y, trainer.rgba_embedding.weight)
        z = zdist.sample((args.batchsize,))
        # Discriminator updates

        dloss, reg = trainer.discriminator_trainstep(x_real, None, z)
        losses["d"].append(dloss)
        # Generators updates
        #        if ((it + 1) % args.d_steps) == 0:
        z = zdist.sample((args.batchsize,))
        gloss = trainer.generator_trainstep(None, z)
        losses["g"].append(gloss)

    return losses


def get_data(args):
    print("loading train data")

    ################# FIXME!!!!
    args.allow_same = False
    args.max_meta = 20
    args.sidelength = 16
    args.expected_input_size = args.sidelength
    args.expected_output_size = args.sidelength
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

    return rDL, train_data


def main(args):
    rDL, train_data = get_data(args)
    print("making models")
    model = models.GAN(args)
    rgba_embedding = models.build_rgba_embed(args.max_meta)
    if args.cuda:
        model.cuda()
        rgba_embedding = rgba_embedding.cuda()
    else:
        print("warning:  no cuda")

    optimizer_d, optimizer_g = get_optimizer(args, model)
    trainer = Trainer(
        model.G,
        model.D,
        optimizer_g,
        optimizer_d,
        gan_type="standard",
        reg_type="real",
        reg_param=args.reg_param,
    )

    trainer.rgba_embedding = rgba_embedding

    args._model = model
    args._rgba_embedding = rgba_embedding
    print("training")
    win_name = None
    for m in range(args.num_epochs):
        losses = train_epoch(model, rDL, trainer, args)
        status = format_stats(
            {
                "epoch": m,
                "gloss": sum(losses["g"]) / len(losses["g"]),
                "dloss": sum(losses["d"]) / len(losses["d"]),
            }
        )
        print(status)
        if args.save_model_dir != "":
            fpath = models.model_filename_from_opts(
                args, savedir=args.save_model_dir, uid=args.save_model_uid
            )
            model.save(fpath)
            save_optimizer(optimizer_d, optimizer_g, fpath)
        if args.visualize:
            zdist = get_zdist("gauss", args.zdim, device=model.G.layers[0].weight.device)
            z = zdist.sample((4,))
            with torch.no_grad():
                u = model.G(z)
            win_name = draw_rgb(u[0], vis, threshold=0.1, win=win_name, title=str(m))

    return model, train_data, rDL


if __name__ == "__main__":
    import argparse
    from voxel_models.plot_voxels import SchematicPlotter, draw_rgb  # noqa
    import visdom

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", type=int, default=-1, help="no shuffle, keep only debug num examples"
    )
    parser.add_argument("--visualize", action="store_true", help="draw pictures")
    parser.add_argument("--num_examples", type=int, default=1024, help="num examples in an epoch")
    parser.add_argument("--num_epochs", type=int, default=1500, help="training epochs")
    parser.add_argument("--last_layer_sigmoid", type=int, default=1, help="do sigmoid if 1")
    parser.add_argument("--num_layers", type=int, default=3, help="numm conv layers")
    # parser.add_argument("--augment", default="none", help="none or maxshift:K_underdirt:J")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--zdim", type=int, default=256, help="hidden variable size")
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
    parser.add_argument(
        "--optim_type", default="rmsprop", help="optim type, rmsprop, adam, adagrad, sgd"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="size of hidden dim")
    parser.add_argument(
        "--load_dictionary",
        default="/private/home/aszlam/junk/word_modify_word_ids.pk",
        help="where to get word dict",
    )
    parser.add_argument("--reg_param", type=float, default=10.0, help="reg_param")
    parser.add_argument("--lr_g", type=float, default=0.0001, help="step size for net")
    parser.add_argument("--lr_d", type=float, default=0.0001, help="step size for net")
    parser.add_argument("--lr_anneal", type=float, default=1.0, help="step multiplier on anneal")
    parser.add_argument("--lr_anneal_every", type=int, default=150000, help="when to anneal")
    parser.add_argument("--d_steps", type=int, default=1, help="when to anneal")
    parser.add_argument(
        "--sbatch", action="store_true", help="cluster run mode, no visdom, formatted stdout"
    )
    parser.add_argument("--ndonkeys", type=int, default=8, help="workers in dataloader")
    args = parser.parse_args()
    if not args.sbatch:
        vis = visdom.Visdom(server="http://localhost")
        sp = SchematicPlotter(vis)

    main(args)
