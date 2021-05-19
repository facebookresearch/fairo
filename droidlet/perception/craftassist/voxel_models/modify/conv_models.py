"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import pickle

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
MC_DIR = os.path.join(THIS_DIR, "../../../../")


def model_filename_from_opts(opts, savedir=None, uid=None):
    filler = "]["
    name = "["
    if opts.ae:
        name += "ae"
        name += filler
    name = name + "hdim" + str(opts.hidden_dim) + filler
    name = name + "edim" + str(opts.embedding_dim) + filler
    #    name = name + "lrp" + str(opts.lr_patience) + filler
    name = name + "lrd" + str(opts.lr_decay) + filler
    name = name + "res" + str(opts.residual_connection) + filler
    name = name + "num_layers" + str(opts.num_layers) + filler
    name = name + "color_io" + str(opts.color_io) + filler
    name = name + "color_hash" + str(opts.color_hash) + filler
    name = name + "sl" + str(opts.sidelength) + filler
    name = name + "sigmoid" + str(opts.last_layer_sigmoid) + filler
    name = name + "lr" + str(opts.lr) + filler
    name = name + opts.optim_type
    if uid is not None and uid != "":
        name = name + filler + uid
    name = name + "].pth"
    if savedir is not None:
        name = os.path.join(savedir, name)
    return name


def get_colors():
    ims = pickle.load(open(os.path.join(MC_DIR, "lowlevel/minecraft/minecraft_specs/block_images/block_data"), "rb"))
    colors = {}
    for b, I in ims["bid_to_image"].items():
        I = I.reshape(1024, 4)
        c = np.zeros(4)
        if not all(I[:, 3] < 0.2):
            c[:3] = I[I[:, 3] > 0.2, :3].mean(axis=0) / 256.0
            c[3] = I[:, 3].mean() / 256.0
        colors[b] = c
    return colors


def build_rgba_embed(max_meta, color_io=2):
    if color_io > 1:
        edim = 4
    else:
        edim = 1
    embedding = nn.Embedding(256 * max_meta, edim)
    embedding.weight.requires_grad = False
    colors = get_colors()
    for b, c in colors.items():
        u = c
        if color_io == 1:
            u = (c.mean(),)
        elif color_io == 0:
            u = (1,)
        bid = b[1] + max_meta * b[0]
        if bid >= 256 * max_meta:
            continue
        embedding.weight[bid][0] = u[0]
        if color_io > 1:
            embedding.weight[bid][1] = u[1]
            embedding.weight[bid][2] = u[2]
            embedding.weight[bid][3] = u[3]
    return embedding


def fake_embedding_fwd(x, embedding_weights):
    embedding_dim = embedding_weights.shape[1]
    szs = list(x.size())
    x = x.view(-1)
    z = embedding_weights.index_select(0, x)
    szs.append(embedding_dim)
    z = z.view(torch.Size(szs))
    z = z.permute(0, 4, 1, 2, 3).contiguous()
    return z


def compressed_onehot_distribution(x, allowed_idxs, pool=False):
    """x is a B x H x W x D LongTensor of indices;
    if not pool, returns a tensor of the same size, with indices mapped to 0:len(allowed_idxs)-1
    if pool, maps to onehot, and pools, returning B x len(allowed_idxs) x H x W x D"""

    k = len(allowed_idxs)
    vals, sidxs = allowed_idxs.sort()
    r = torch.arange(0, len(vals), dtype=allowed_idxs.dtype, device=allowed_idxs.device)
    u = torch.zeros(vals[-1].item() + 1, dtype=allowed_idxs.dtype, device=allowed_idxs.device)
    u[vals] = r
    mapped_x = u[x]
    if pool:
        weight = torch.eye(k, device=x.device)
        onehot = fake_embedding_fwd(mapped_x, weight)
        return torch.nn.functional.avg_pool3d(onehot, pool, stride=pool)
    else:
        return mapped_x


def color_hash(x, nbins=3):
    # x is assumed to be Nx4, and each entry is 0<=x[i]<= 1
    q = (x[:, :3] * (nbins - 0.001)).floor().to(dtype=torch.long)
    b = x[:, 3] < 0.02
    q[b] = 0
    b = 1 - b.to(dtype=torch.long)
    return b + q[:, 0] * nbins ** 2 + q[:, 1] * nbins + q[:, 2]


class ConvNLL(nn.Module):
    def __init__(self, max_meta=20, subsample_zeros=-1):
        super(ConvNLL, self).__init__()
        self.embedding = build_rgba_embed(max_meta, color_io=2)
        self.subsample_zeros = subsample_zeros
        self.nll = nn.NLLLoss()
        self.lsm = nn.LogSoftmax()

    def cuda(self):
        self.embedding.cuda()

    def forward(self, gold, scores, nbins):
        gold = gold.view(-1)
        embedded_gold = self.embedding.weight.index_select(0, gold)
        hashed_eg = color_hash(embedded_gold, nbins=nbins)
        if self.subsample_zeros > 0:
            mask = (hashed_eg == 0).float()
            n = torch.rand(hashed_eg.shape[0], device=mask.device)
            mask = mask - n
            keep_nz_idx = torch.nonzero(mask < self.subsample_zeros).view(-1)

        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        szs = list(scores.size())
        scores = scores.view(-1, szs[-1])
        if self.subsample_zeros > 0:
            scores = scores[keep_nz_idx]
            hashed_eg = hashed_eg[keep_nz_idx]
        return self.nll(self.lsm(scores), hashed_eg)


""" does a batch nce over B x c x H x W x D
    draws negatives from self.embedder
"""


class ConvDistributionMatch(nn.Module):
    def __init__(self, embedding, pool=False, subsample_zeros=-1):
        super(ConvDistributionMatch, self).__init__()
        self.pool = pool
        self.embedding = embedding
        self.K = self.embedding.weight.shape[0]
        self.lsm = nn.LogSoftmax(dim=1)
        self.lsm.to(self.embedding.weight.device)
        self.subsample_zeros = subsample_zeros

    def forward(self, gold, z, allowed_idxs):
        pool = self.pool
        mapped_gold = compressed_onehot_distribution(gold, allowed_idxs, pool=pool)
        if not pool:
            mapped_gold = mapped_gold.view(-1)
        else:
            mapped_gold = mapped_gold.permute(0, 2, 3, 4, 1).contiguous()
            szs = list(mapped_gold.size())
            mapped_gold = mapped_gold.view(-1, szs[-1])
            self.mapped_gold = mapped_gold

        # FIXME will break with pool
        if self.subsample_zeros > 0:
            mask = (mapped_gold == 0).float()
            n = torch.rand(mapped_gold.shape[0], device=mask.device)
            mask = mask - n
            keep_nz_idx = torch.nonzero(mask < self.subsample_zeros).view(-1)

        weight = self.embedding.weight.index_select(0, allowed_idxs)
        k = weight.shape[0]
        d = weight.shape[1]
        scores = nn.functional.conv3d(z, weight.view(k, d, 1, 1, 1))
        self.scores = scores
        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        szs = list(scores.size())
        scores = scores.view(-1, szs[-1])
        if self.subsample_zeros > 0:
            scores = scores[keep_nz_idx]
            mapped_gold = mapped_gold[keep_nz_idx]
        if pool:
            kl = nn.KLDivLoss()
            return kl(self.lsm(scores), mapped_gold)
        else:
            #            nll_weight = torch.ones(len(allowed_idxs), device=weight.device)
            #            nll_weight[0] = 0.01
            #            nll = nn.NLLLoss(weight=nll_weight)
            nll = nn.NLLLoss()
            return nll(self.lsm(scores), mapped_gold)


# this will need ot be fixed when we have relative directions!!!!
class SimpleWordEmbedder(nn.Module):
    def __init__(self, opts):
        super(SimpleWordEmbedder, self).__init__()
        self.embedding = nn.Embedding(
            opts.num_words, opts.hidden_dim, padding_idx=opts.word_padding_idx
        )

    def forward(self, words):
        return self.embedding(words).mean(1)


class SimpleBase(nn.Module):
    def __init__(self, opts, filepath=None):
        super(SimpleBase, self).__init__()
        self.loaded_from = None
        if not filepath and opts.load_model_dir != "":
            filepath = model_filename_from_opts(
                opts, savedir=opts.load_model_dir, uid=opts.save_model_uid
            )
        if filepath:
            try:
                self.load(filepath)
                self.loaded_from = filepath
            except:
                if opts.load_strict:
                    raise ("tried to load from " + filepath + " but failed")
                else:
                    print("warning:  tried to load from " + filepath + " but failed")
                    print("starting new model")
                    self.opts = opts
                    self._build()
        else:
            self.opts = opts
            self._build()

    def _build(self):
        pass

    def save(self, filepath):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["state_dict"] = self.state_dict()
        torch.save(sds, filepath)
        if self.opts.cuda:
            self.cuda()

    def load(self, filepath):
        sds = torch.load(filepath)
        self.opts = sds["opts"]
        print("loading from file, using opts")
        print(self.opts)
        self._build()
        self.load_state_dict(sds["state_dict"])
        self.zero_grad()


class SimpleConv(SimpleBase):
    def __init__(self, opts, pool=False):
        opts.pool = pool
        super(SimpleConv, self).__init__(opts)

    def _build(self):
        opts = self.opts
        if hasattr(opts, "pool"):
            self.pool = opts.pool
        else:
            self.pool = None
        self.max_meta = max(opts.max_meta, 20)
        self.num_blocks = 256 * self.max_meta
        num_blocks = self.num_blocks
        embedding_dim = opts.embedding_dim
        num_layers = opts.num_layers
        hidden_dim = opts.hidden_dim

        self.embedding_dim = embedding_dim
        if opts.color_io >= 0:
            self.embedding = build_rgba_embed(self.max_meta, color_io=opts.color_io)
            self.embedding_dim = self.embedding.weight.shape[1]
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_blocks, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(self.embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        self.gate_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

        if self.opts.color_hash > 0:
            self.out = nn.Conv3d(hidden_dim, self.opts.color_hash ** 3 + 1, kernel_size=1)
        else:
            self.out = nn.Conv3d(hidden_dim, self.embedding_dim, kernel_size=1)

        self.lvar_embedder = nn.Embedding(opts.num_lvars, hidden_dim)
        self.words_embedder = SimpleWordEmbedder(opts)

    # TODO attention everywhere...
    def forward(self, blocks_array, words, lvars):
        words_embeddings = self.words_embedder(words)
        lvar_embeddings = self.lvar_embedder(lvars)
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        z = fake_embedding_fwd(blocks_array, self.embedding.weight)
        if self.pool:
            z = torch.nn.functional.avg_pool3d(z, self.pool, stride=self.pool)
        # words_embeddings should be a batchsize x hidden_dim vector
        #        z = z + words_embeddings.view(wszs).expand(szs)
        #        z = z + lvar_embeddings.view(wszs).expand(szs)
        for i in range(self.num_layers):
            oz = z.clone()
            z = self.layers[i](z)
            szs = list(z.size())
            wszs = szs.copy()
            wszs[2] = 1
            wszs[3] = 1
            wszs[4] = 1
            z = z + words_embeddings.view(wszs).expand(szs)
            z = z + lvar_embeddings.view(wszs).expand(szs)
            if self.opts.residual_connection > 0 and oz.shape[1] == z.shape[1]:
                z = z + oz
        return self.out(z)


class AE(SimpleBase):
    def __init__(self, opts, filepath=None):
        super(AE, self).__init__(opts, filepath=filepath)

    def _build(self):
        opts = self.opts
        self.do_sigmoid = opts.last_layer_sigmoid == 1
        self.max_meta = max(opts.max_meta, 20)
        self.num_blocks = 256 * self.max_meta
        num_blocks = self.num_blocks
        embedding_dim = opts.embedding_dim
        num_layers = opts.num_layers
        if opts.color_io >= 0:
            self.embedding = build_rgba_embed(self.max_meta, color_io=opts.color_io)
            self.embedding_dim = self.embedding.weight.shape[1]
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_blocks, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        current_dim = self.embedding_dim
        for i in range(num_layers):
            if i == 0:
                hdim = self.opts.hidden_dim
            else:
                hdim = 2 * current_dim
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(current_dim, hdim, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm3d(hdim),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = hdim
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        current_dim,
                        current_dim // 2,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.BatchNorm3d(current_dim // 2),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = current_dim // 2
        if self.opts.color_hash > 0:
            self.pre_out = nn.Conv3d(current_dim, self.opts.color_hash ** 3 + 1, kernel_size=1)
        else:
            self.pre_out = nn.Conv3d(current_dim, self.embedding_dim, kernel_size=1)

    # TODO attention everywhere...
    def forward(self, blocks_array):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        z = fake_embedding_fwd(blocks_array, self.embedding.weight)
        self.input_embed = z.clone()
        for i in range(self.num_layers):
            z = self.layers[i](z)
        self.hidden_state = z
        for i in range(self.num_layers, 2 * self.num_layers):
            z = self.layers[i](z)
        z = self.pre_out(z)
        if self.do_sigmoid and self.opts.color_hash < 0:
            return torch.sigmoid(z)
        else:
            return z


class ConvGenerator(nn.Module):
    def __init__(self, opts):
        super(ConvGenerator, self).__init__()
        self.opts = opts
        self.hidden_dim = opts.hidden_dim
        self.zdim = opts.zdim
        self.do_sigmoid = opts.last_layer_sigmoid == 1
        self.layers = nn.ModuleList()
        self.num_layers = opts.num_layers
        self.expected_output_size = opts.expected_output_size
        self.base_grid = opts.expected_output_size // 2 ** self.num_layers
        current_dim = self.hidden_dim
        self.layers.append(nn.Linear(self.zdim, self.hidden_dim * self.base_grid ** 3))

        for i in range(self.num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        current_dim,
                        current_dim // 2,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.BatchNorm3d(current_dim // 2),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = current_dim // 2
        self.pre_out = nn.Conv3d(current_dim, 4, kernel_size=1)

    def forward(self, z, c=None):
        z = self.layers[0](z)
        szs = z.shape
        z = z.view(szs[0], -1, self.base_grid, self.base_grid, self.base_grid)
        for i in range(self.num_layers):
            z = self.layers[i + 1](z)
        z = self.pre_out(z)
        if self.do_sigmoid:
            return torch.sigmoid(z)
        else:
            return z


class ConvDiscriminator(nn.Module):
    def __init__(self, opts):
        super(ConvDiscriminator, self).__init__()
        self.opts = opts
        self.zdim = opts.zdim
        self.do_sigmoid = opts.last_layer_sigmoid == 1
        self.layers = nn.ModuleList()
        self.num_layers = opts.num_layers
        self.expected_input_size = opts.expected_input_size

        self.layers = nn.ModuleList()

        current_dim = 4  # RGBA
        for i in range(self.num_layers):
            if i == 0:
                hdim = self.opts.hidden_dim
            else:
                hdim = 2 * current_dim
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(current_dim, hdim, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm3d(hdim),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = hdim

        self.base_grid = opts.expected_input_size // 2 ** self.num_layers
        self.pre_out = nn.Linear(current_dim * self.base_grid ** 3, 1)

    def forward(self, z, c=None):
        for i in range(self.num_layers):
            z = self.layers[i](z)
        z = z.view(z.shape[0], -1)
        z = self.pre_out(z)
        return torch.tanh(z)


class GAN(SimpleBase):
    def __init__(self, opts):
        super(GAN, self).__init__(opts)

    def _build(self):
        self.D = ConvDiscriminator(self.opts)
        self.G = ConvGenerator(self.opts)

    def forward(self, x, mode="D"):
        if mode == "D":
            return self.D(x)
        else:
            return self.G(x)


class Opt:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=16, help="size of blockid embedding")
    parser.add_argument("--num_words", type=int, default=256, help="number of blocks")
    parser.add_argument("--num_classes", type=int, default=20, help="number of blocks")

    args = parser.parse_args()

#    N = SemSegNet(args)
