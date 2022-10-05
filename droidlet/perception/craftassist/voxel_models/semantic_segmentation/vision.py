"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import torch.nn as nn
from droidlet.perception.craftassist.voxel_models.semantic_segmentation.data_loaders import make_example_from_raw

from transformers import DistilBertTokenizer, DistilBertModel

from droidlet.lowlevel.minecraft.small_scenes_with_shapes import SL, H

import clip

BERT_HIDDEN_DIM = 768
CLIP_HIDDEN_DIM = 512

class SemSegNet(nn.Module):
    """Semantic Segmentation Neural Network"""

    def __init__(self, opts, classes=None):
        super(SemSegNet, self).__init__()
        if opts.load:
            if opts.load_model != "":
                print(f"Loading pretrained semseg model...")
                self.load(opts.load_model)
            else:
                raise ("loading from file specified but no load_filepath specified")
        else:
            self.opts = opts
            self._build()
            self.classes = classes

    def _build(self):
        opts = self.opts
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 8
        try:
            num_words = opts.num_words
        except:
            num_words = 3
        try:
            num_layers = opts.num_layers
        except:
            num_layers = 4 
        try:
            hidden_dim = opts.hidden_dim
        except:
            hidden_dim = 64

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        if opts.query_embed == "lut":
            print(f"Using lut as query embedding")
            TEXT_HIDDEN_DIM = 1
        elif opts.query_embed == "bert":
            print(f"Using BERT as query embedding")
            TEXT_HIDDEN_DIM = BERT_HIDDEN_DIM
        elif opts.query_embed == "clip":
            TEXT_HIDDEN_DIM = CLIP_HIDDEN_DIM
        else:
            raise Exception(f"Invalid Query Embedding: {opts.query_embed}!")
        self.text_proj = nn.Linear(TEXT_HIDDEN_DIM, hidden_dim)
        # self.linear = nn.Linear(hidden_dim + BERT_HIDDEN_DIM, 1)
        # self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        szs = list(x.size()) # B x SL x SL x SL
        B = szs[0]
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()

        for i in range(self.num_layers):
            z = self.layers[i](z)
        t = self.text_proj(t) # B x Q x H
        t = t.permute(0, 2, 1) # B x H x Q
        z = z.permute(0, 2, 3, 4, 1).view(B, -1, self.opts.hidden_dim) # B x (SL x SL x SL) x H
        z = torch.bmm(z, t).view(B, szs[1], szs[2], szs[3], -1) # B x (SL x SL x SL) x Q
        # do a contrastive not yet
        # multiple tags for one scene
        z = self.sigmoid(z)
        # print(f"z size: {z.size()}")
        return z


    def save(self, filepath):
        # self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["classes"] = self.classes
        sds["state_dict"] = self.state_dict()
        torch.save(sds, filepath)
        # if self.opts.cuda:
        #     self.cuda()

    def load(self, filepath):
        sds = torch.load(filepath)
        self.opts = sds["opts"]
        print("loading from file, using opts")
        print(self.opts)
        self._build()
        self.load_state_dict(sds["state_dict"], strict=False)
        self.zero_grad()
        self.classes = sds["classes"]


class Opt:
    pass


class SemSegWrapper:
    """Wrapper for Semantic Segmentation Net"""

    def __init__(self, model, threshold=0.5, blocks_only=True, cuda=False):
        if type(model) is str:
            opts = Opt()
            opts.load = True
            opts.load_model = model
            model = SemSegNet(opts)
        self.model = model
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        self.classes = model.classes
        # threshold for relevance; unused rn
        if model.opts.prob_threshold:
            self.threshold = model.opts.prob_threshold
        else:
            self.threshold = threshold
        # if true only label non-air blocks
        self.blocks_only = blocks_only
        # this is used by the semseg_process
        i2n = self.classes["idx2name"]
        self.tags = [(c, self.classes["name2count"][c]) for c in i2n]
        # assert self.classes["name2idx"]["none"] == 0

        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)

        self.device = "cuda"#opts.device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    @torch.no_grad()
    def segment_object(self, blocks):
        self.model.eval()
        blocks = torch.from_numpy(blocks)[:, :, :, 0]
        blocks, _, o = make_example_from_raw(blocks)
        blocks = blocks.unsqueeze(0)
        if self.cuda:
            blocks = blocks.cuda()
        y = self.model(blocks)
        _, mids = y.squeeze().max(0)
        locs = mids.nonzero()
        locs = locs.tolist()
        if self.blocks_only:
            return {
                tuple(np.subtract(l, o)): mids[l[0], l[1], l[2]].item()
                for l in locs
                if blocks[0, l[0], l[1], l[2]] > 0
            }
        else:
            return {tuple(ll for ll in l): mids[l[0], l[1], l[2]].item() for l in locs}
        
    
    def encode_text(self, text, embed_type="clip"):
        if embed_type == "bert":
            text_inputs = self.bert_tokenizer(text, return_tensors="pt")
            text_outputs = self.bert_model(**text_inputs)
            last_hidden_state = text_outputs.last_hidden_state
            text_embed = torch.squeeze(torch.sum(last_hidden_state, 1))
        elif embed_type == "clip":
            with torch.no_grad():
                tokenized_text = clip.tokenize(text).to(self.device)
                text_embed = self.clip_model.encode_text(tokenized_text).float()
        return text_embed

    @torch.no_grad()
    def perceive(self, blocks, text):
        text_embed = self.encode_text(text)
        self.model.eval()
        blocks = torch.from_numpy(blocks).long()
        xmax, ymax, zmax = blocks.size()
        print(f"xmax, ymax, zmax: {xmax}, {ymax}, {zmax}")
        # blocks, _, o = make_example_from_raw(blocks)
        # print(o)
        blocks = blocks.unsqueeze(0)
        if self.cuda:
            blocks = blocks.cuda()
            text_embed = text_embed.cuda()
        text_embed = text_embed.unsqueeze(0)
        y = self.model(blocks, text_embed)
        preds = y > self.threshold
        ret = []
        for i in range(preds.size(4)):
            pred = preds[:, :, :, :, i].squeeze()
            locs = pred.nonzero()
            res = [tuple(l) for l in locs]
            p = []
            for loc in res:
                x, y, z = loc
                if x >= 0 and x < xmax and y >= 0 and y < ymax and z >= 0 and z < zmax:
                    p.append((int(x), int(y), int(z)))
            ret.append(p)
        return ret



        # locs = pred.squeeze().nonzero()
        # locs = locs.tolist()
        # print(len(locs))
        # res = [tuple(l) for l in locs]
        # pred = []
        # for loc in res:
        #     x, y, z = loc
        #     if x >= 0 and x < xmax and y >= 0 and y < ymax and z >= 0 and z < zmax:
        #         pred.append((int(x), int(y), int(z)))
        # print(len(pred))
        # return pred


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

    N = SemSegNet(args)
