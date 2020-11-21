"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import directional_utils as du


def conv3x3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv3x3x3up(in_planes, out_planes, bias=True):
    """3x3x3 convolution with padding"""
    return nn.ConvTranspose3d(
        in_planes, out_planes, stride=2, kernel_size=3, padding=1, output_padding=1
    )


def convbn(in_planes, out_planes, stride=1, bias=True):
    return nn.Sequential(
        (conv3x3x3(in_planes, out_planes, stride=stride, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


def convbnup(in_planes, out_planes, bias=True):
    return nn.Sequential(
        (conv3x3x3up(in_planes, out_planes, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


class ContextEmbeddingNet(nn.Module):
    def __init__(self, opts, blockid_embedding):
        super(ContextEmbeddingNet, self).__init__()

        self.blockid_embedding_dim = opts.get("blockid_embedding_dim", 8)
        output_embedding_dim = opts.get("output_embedding_dim", 8)
        num_layers = opts.get("num_layers", 4)
        hidden_dim = opts.get("hidden_dim", 64)
        self.use_direction = opts.get("cont_use_direction", False)
        self.use_xyz_from_viewer_look = opts.get("cont_use_xyz_from_viewer_look", False)
        self.c_sl = opts.get("context_side_length", 32)
        self.xyz = None

        input_dim = self.blockid_embedding_dim
        if self.use_direction:
            input_dim += 5
        if self.use_xyz_from_viewer_look:
            input_dim += 3
            self.xyz = du.create_xyz_tensor(self.c_sl).view(1, -1, 3)
            # self.viewer_look = du.get_viewer_look(self.c_sl)
            if opts.get("cuda", 0):
                self.xyz = self.xyz.cuda()
                # self.viewer_look = self.viewer_look.cuda()

        # A shared embedding for the block id types
        self.blockid_embedding = blockid_embedding

        # Create model for converting the context into HxWxL D dim representations
        self.layers = nn.ModuleList()
        # B dim block id -> hidden dim, maintain input size
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        # hidden dim -> hidden dim, maintain input size
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        # hidden dim -> spatial embedding dim, maintain input size
        self.out = nn.Linear(hidden_dim, output_embedding_dim)

    # Input: [context, opt:viewer_pos, opt:viewer_look, opt:direction]
    # Returns N x D x H x W x L
    def forward(self, b):
        bsls = b["context"].size()[1:]
        if bsls[0] != self.c_sl or bsls[1] != self.c_sl or bsls[2] != self.c_sl:
            raise Exception(
                "Size of context should be Nx{}x{}x{} but it is {}".format(
                    self.c_sl, self.c_sl, self.c_sl, b["context"].size()
                )
            )

        sizes = list(b["context"].size())
        x = b["context"].view(-1)
        # Get the blockid embedding for each space in the context input
        z = self.blockid_embedding.weight.index_select(0, x)
        z = z.float()
        # Add the embedding dim B
        sizes.append(self.blockid_embedding_dim)

        # z: N*D x B
        if self.use_xyz_from_viewer_look:
            n_xyz = self.xyz.expand(sizes[0], -1, -1)
            # Input: viewer pos, viewer look (N x 3), n_xyz (N x D x 3)
            n_xyz = (
                du.get_xyz_viewer_look_coords_batched(b["viewer_pos"], b["viewer_look"], n_xyz)
                .view(-1, 3)
                .float()
            )
            z = torch.cat([z, n_xyz], 1)
            # Add the xyz_look_position to the input size list
            sizes[-1] += 3

        if self.use_direction:
            # direction: N x 5
            direction = b["dir_vec"]
            d = self.c_sl * self.c_sl * self.c_sl
            direction = direction.unsqueeze(1).expand(-1, d, -1).contiguous().view(-1, 5)
            direction = direction.float()
            z = torch.cat([z, direction], 1)
            # Add the direction emb to the input size list
            sizes[-1] += 5

        z = z.view(torch.Size(sizes))
        # N x H x W x L x B ==> N x B x H x W x L
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        return self.out(z)


class SegmentEmbeddingNet(nn.Module):
    def __init__(self, opts, blockid_embedding):
        super(SegmentEmbeddingNet, self).__init__()

        self.blockid_embedding_dim = opts.get("blockid_embedding_dim", 8)
        spatial_embedding_dim = opts.get("spatial_embedding_dim", 8)
        hidden_dim = opts.get("hidden_dim", 64)
        self.s_sl = 8  # TODO make this changeable in model arch

        # A shared embedding for the block id types
        self.blockid_embedding = blockid_embedding

        # Create model for converting the segment into 1 D dim representation
        # input size: 8x8x8
        self.layers = nn.ModuleList()
        # B dim block id -> hidden dim, maintain input size
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(self.blockid_embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        # hidden dim -> hidden dim
        #   (maintain input size x2, max pool to half) x 3: 8x8x8 ==> 1x1x1
        for i in range(3):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2, stride=2),
                )
            )
        # hidden dim -> spatial embedding dim, 1x1x1
        self.out = nn.Linear(hidden_dim, spatial_embedding_dim)

    # Returns N x D x 1 x 1 x 1
    def forward(self, b):
        bsls = b["seg"].size()[1:]
        if bsls[0] != self.s_sl or bsls[1] != self.s_sl or bsls[2] != self.s_sl:
            raise Exception("Size of input should be Nx8x8x8 but it is {}".format(b["seg"].size()))
        sizes = list(b["seg"].size())
        seg = b["seg"].view(-1)
        # Get the blockid embedding for each space in the context input
        z = self.blockid_embedding.weight.index_select(0, seg)
        # Add the embedding dim B
        sizes.append(self.blockid_embedding_dim)
        z = z.view(torch.Size(sizes))
        # N x H x W x L x B ==> N x B x H x W x L
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        return self.out(z)


class SegmentDirectionEmbeddingNet(nn.Module):
    def __init__(self, opts):
        super(SegmentDirectionEmbeddingNet, self).__init__()

        output_embedding_dim = opts.get("output_embedding_dim", 8)
        self.use_viewer_pos = opts.get("seg_use_viewer_pos", False)
        self.use_direction = opts.get("seg_use_direction", False)
        hidden_dim = opts.get("hidden_dim", 64)
        num_layers = opts.get("num_seg_dir_layers", 3)
        self.seg_input_dim = opts.get("spatial_embedding_dim", 8)
        self.c_sl = opts.get("context_side_length", 32)
        input_dim = self.seg_input_dim
        if self.use_viewer_pos:
            input_dim += 3
        if self.use_direction:
            input_dim += 5

        # Create model for converting the segment, viewer info,
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.out = nn.Linear(hidden_dim, output_embedding_dim)

    # In: batch dict, must have s_embeds, viewer_pos, dir_vec
    # Out: N x D x 1 x 1 x 1
    def forward(self, b):
        if b["s_embeds"].size()[1] != self.seg_input_dim:
            raise Exception("The seg spatial embed is wrong size: {}".format(b["s_embeds"].size()))

        inp = [b["s_embeds"]]
        normalizing_const = self.c_sl * 1.0 / 2.0
        if self.use_viewer_pos:
            inp.append(b["viewer_pos"].float().div_(normalizing_const))
        if self.use_direction:
            inp.append(b["dir_vec"].float())

        z = torch.cat(inp, 1)
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return self.out(z).unsqueeze(2).unsqueeze(3).unsqueeze(4)


class ContextSegmentScoringModule(nn.Module):
    def __init__(self):
        super(ContextSegmentScoringModule, self).__init__()

    def forward(self, x):
        context_emb = x["c_embeds"]  # N x 32 x 32 x 32 x D
        seg_emb = x["s_embeds"]  # N x 1 x 1 x 1 x D

        c_szs = context_emb.size()  # N x 32 x 32 x 32 x D
        batch_dim = c_szs[0]
        emb_dim = c_szs[4]
        num_scores = c_szs[1] * c_szs[2] * c_szs[3]

        # Prepare context for the dot product
        context_emb = context_emb.view(-1, emb_dim, 1)  # N*32^3 x D x 1

        # Prepare segment for the dot product
        seg_emb = seg_emb.view(batch_dim, 1, -1)  # N x 1 x D
        seg_emb = seg_emb.expand(-1, num_scores, -1).contiguous()  # N x 32^3 x D
        seg_emb = seg_emb.view(-1, 1, emb_dim)  # N*32^3 x 1 x D

        # Dot product & reshape
        # (K x 1 x D) bmm (K x D x 1) = (K x 1 x 1)
        out = torch.bmm(seg_emb, context_emb)
        return out.view(batch_dim, -1)


class spatial_emb_loss(nn.Module):
    def __init__(self):
        super(spatial_emb_loss, self).__init__()
        self.lsm = nn.LogSoftmax()
        self.crit = nn.NLLLoss()

    # format [scores (Nx32^3), targets (N)]
    def forward(self, inp):
        assert len(inp) == 2
        scores = inp[0]
        targets = inp[1]
        logsuminp = self.lsm(scores)
        return self.crit(logsuminp, targets)


class rank_loss(nn.Module):
    def __init__(self, margin=0.1, nneg=5):
        super(rank_loss, self).__init__()
        self.nneg = 5
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, inp):
        # it is expected that the batch is arranged as pos neg neg ... neg pos neg ...
        # with self.nneg negs per pos
        assert inp.shape[0] % (self.nneg + 1) == 0
        inp = inp.view(self.nneg + 1, -1)
        pos = inp[0]
        neg = inp[1:].contiguous()
        errors = self.relu(neg - pos.repeat(self.nneg, 1) + self.margin)
        return errors.mean()


class reshape_nll(nn.Module):
    def __init__(self, nneg=5):
        super(reshape_nll, self).__init__()
        self.nneg = nneg
        self.lsm = nn.LogSoftmax()
        self.crit = nn.NLLLoss()

    def forward(self, inp):
        # it is expected that the batch is arranged as pos neg neg ... neg pos neg ...
        # with self.nneg negs per pos
        assert inp.shape[0] % (self.nneg + 1) == 0
        inp = inp.view(-1, self.nneg + 1).contiguous()
        logsuminp = self.lsm(inp)
        o = torch.zeros(inp.size(0), device=inp.device).long()
        return self.crit(logsuminp, o)


def get_optim(model_params, opts):
    optim_type = opts.get("optim", "adagrad")
    lr = opts.get("lr", 0.1)
    momentum = opts.get("momentum", 0.0)
    betas = (0.9, 0.999)

    if optim_type == "adagrad":
        return optim.Adagrad(model_params, lr=lr)
    elif optim_type == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=momentum)
    elif optim_type == "adam":
        return optim.Adam(model_params, lr=lr, betas=betas)
    else:
        raise Exception("Undefined optim type {}".format(optim_type))


def create_context_segment_modules(opts):
    possible_params = ["context_net", "seg_net", "seg_direction_net"]

    # Add all of the modules
    emb_dict = torch.nn.Embedding(opts["num_words"], opts["blockid_embedding_dim"])
    tms = {
        "context_net": ContextEmbeddingNet(opts, emb_dict),
        "seg_net": SegmentEmbeddingNet(opts, emb_dict),
        "score_module": ContextSegmentScoringModule(),
        "lfn": spatial_emb_loss(),
    }
    if opts.get("seg_direction_net", False):
        tms["seg_direction_net"] = SegmentDirectionEmbeddingNet(opts)

    # Move everything to the right device
    if "cuda" in opts and opts["cuda"]:
        emb_dict.cuda()
        for n in possible_params:
            if n in tms:
                tms[n].cuda()

    # Setup the optimizer
    all_params = []
    for n in possible_params:
        if n in tms:
            all_params.extend(list(tms[n].parameters()))
    tms["optimizer"] = get_optim(all_params, opts)
    return tms
