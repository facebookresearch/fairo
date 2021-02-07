"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, Adagrad

from transformers.modeling_bert import BertModel, BertOnlyMLMHead

from utils_caip import *


def my_xavier_init(m, gain=1):
    """Xavier initialization: weights initialization that tries to make variance of outputs
    of a layer equal to variance of its inputs.
    """
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain)
        else:
            nn.init.constant_(p, 0)


class HighwayLayer(torch.nn.Module):
    """Highway transformation used in span prediction."""

    def __init__(self, dim):
        super(HighwayLayer, self).__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        self.nlin_proj = nn.Linear(dim, dim, bias=True)
        my_xavier_init(self.nlin_proj)
        my_xavier_init(self.gate_proj)
        nn.init.constant_(self.gate_proj.bias, -1)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        nlin = torch.tanh(self.nlin_proj(x))
        res = gate * nlin + (1 - gate) * x
        return res


# single module to predict the output sequence and compute the
# loss if the target sequence is provided for convenience
class DecoderWithLoss(nn.Module):
    """
    Transformer-based decoder module for sequence and span prediction.
    Predicts the output sequence and computes the loss if the target sequence is provided for convenience.

    Attributes:
        bert: BERT model, initialized with config
        lm_head: language modeling head
        span_b_proj: span head predicting span beginning
        span_e_proj: span head predicting span end
        text_span_start_head: text span head predicting start
        text_span_end_head: text span head predicting end
        span_ce_loss: Cross Entropy loss for spans
        span_loss_lb: Label smoothing loss for spans
        text_span_loss: Cross Entropy loss for text spans

    """

    def __init__(self, config, args, tokenizer):
        super(DecoderWithLoss, self).__init__()
        # model components
        print("initializing decoder with params {}".format(args))
        self.bert = BertModel(config)
        self.lm_head = BertOnlyMLMHead(config)
        self.span_b_proj = nn.ModuleList(
            [HighwayLayer(config.hidden_size) for _ in range(args.num_highway)]
        )
        self.span_e_proj = nn.ModuleList(
            [HighwayLayer(config.hidden_size) for _ in range(args.num_highway)]
        )
        # predict text span beginning and end
        self.text_span_start_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.text_span_end_head = nn.Linear(config.hidden_size, config.hidden_size)
        # loss functions
        if args.node_label_smoothing > 0:
            self.lm_ce_loss = LabelSmoothingLoss(
                args.node_label_smoothing, config.vocab_size, ignore_index=tokenizer.pad_token_id
            )
        else:
            self.lm_ce_loss = torch.nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id, reduction="none"
            )
        self.span_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.span_loss_lb = args.lambda_span_loss
        self.text_span_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.tree_to_text = args.tree_to_text

    def step(self, y, y_mask, x_reps, x_mask):
        """Without loss, used at prediction time.

        TODO: add previously computed y_rep, currently y only has the node indices, not spans.

        Args:
            y: targets
            y_mask: mask for targets
            x_reps: encoder hidden states
            x_mask: input mask

        Returns:
            Dictionary containing scores from each output head

        """
        y_rep = self.bert(
            input_ids=y,
            attention_mask=y_mask,
            encoder_hidden_states=x_reps,
            encoder_attention_mask=x_mask,
        )[0]
        y_mask_target = y_mask
        lm_scores = self.lm_head(y_rep)
        y_span_pre_b = y_rep
        for hw in self.span_b_proj:
            y_span_pre_b = hw(y_span_pre_b)
        span_b_scores = (x_reps[:, None, :, :] * y_span_pre_b[:, :, None, :]).sum(dim=-1)
        span_b_scores = (
            span_b_scores + (1 - y_mask_target.type_as(span_b_scores))[:, :, None] * 1e9
        )
        y_span_pre_e = y_rep
        for hw in self.span_e_proj:
            y_span_pre_e = hw(y_span_pre_e)
        span_e_scores = (x_reps[:, None, :, :] * y_span_pre_e[:, :, None, :]).sum(dim=-1)
        span_e_scores = (
            span_e_scores + (1 - y_mask_target.type_as(span_e_scores))[:, :, None] * 1e9
        )
        # text span prediction
        # detach head
        text_span_start_hidden_z = y_rep.detach()
        text_span_end_hidden_z = y_rep.detach()
        # get predicted values
        text_span_start_out = self.text_span_start_head(text_span_start_hidden_z)
        text_span_start_scores = (x_reps[:, None, :, :] * text_span_start_out[:, :, None, :]).sum(
            dim=-1
        )
        text_span_start_scores = (
            text_span_start_scores
            + (1 - y_mask_target.type_as(text_span_start_scores))[:, :, None] * 1e9
        )
        # text span end prediction
        text_span_end_out = self.text_span_end_head(text_span_end_hidden_z)
        text_span_end_scores = (x_reps[:, None, :, :] * text_span_end_out[:, :, None, :]).sum(
            dim=-1
        )
        text_span_end_scores = (
            text_span_end_scores
            + (1 - y_mask_target.type_as(text_span_end_scores))[:, :, None] * 1e9
        )
        res = {
            "lm_scores": torch.log_softmax(lm_scores, dim=-1).detach(),
            "span_b_scores": torch.log_softmax(span_b_scores, dim=-1).detach(),
            "span_e_scores": torch.log_softmax(span_e_scores, dim=-1).detach(),
            "text_span_start_scores": torch.log_softmax(text_span_start_scores, dim=-1).detach(),
            "text_span_end_scores": torch.log_softmax(text_span_end_scores, dim=-1).detach(),
        }
        return res

    def forward(self, y, y_mask, x_reps, x_mask, is_eval=False):
        """Same as step, except with loss. Set is_eval=True for validation."""
        if self.tree_to_text:
            bert_model = self.bert(
                input_ids=y,
                attention_mask=y_mask,
                encoder_hidden_states=x_reps,
                encoder_attention_mask=x_mask,
            )
            y_rep = bert_model[0]
            y_mask_target = y_mask.contiguous()
            # language modeling
            lm_scores = self.lm_head(y_rep)
            lm_lin_scores = lm_scores.view(-1, lm_scores.shape[-1])
            lm_lin_targets = y.contiguous().view(-1)
            lm_lin_loss = self.lm_ce_loss(lm_lin_scores, lm_lin_targets)
            lm_lin_mask = y_mask_target.view(-1)
            res = {"lm_scores": lm_scores, "loss": lm_loss}
        else:
            model_out = self.bert(
                input_ids=y[:, :-1, 0],
                attention_mask=y_mask[:, :-1],
                encoder_hidden_states=x_reps,
                encoder_attention_mask=x_mask,
            )
            y_rep = model_out[0]
            if not is_eval:
                y_rep.retain_grad()
            self.bert_final_layer_out = y_rep
            y_mask_target = y_mask[:, 1:].contiguous()
            # language modeling
            lm_scores = self.lm_head(y_rep)
            lm_lin_scores = lm_scores.view(-1, lm_scores.shape[-1])
            lm_lin_targets = y[:, 1:, 0].contiguous().view(-1)
            lm_lin_loss = self.lm_ce_loss(lm_lin_scores, lm_lin_targets)
            lm_lin_mask = y_mask_target.view(-1)
            lm_loss = lm_lin_loss.sum() / lm_lin_mask.sum()
            lm_loss = lm_lin_loss.sum() / lm_lin_mask.sum()
            # span prediction
            ## beginning of spans
            y_span_pre_b = y_rep
            for hw in self.span_b_proj:
                y_span_pre_b = hw(y_span_pre_b)
            span_b_scores = (x_reps[:, None, :, :] * y_span_pre_b[:, :, None, :]).sum(dim=-1)
            span_b_scores = (
                span_b_scores + (1 - y_mask_target.type_as(span_b_scores))[:, :, None] * 1e9
            )
            span_b_lin_scores = span_b_scores.view(-1, x_reps.shape[1])
            span_b_lin_targets = y[:, 1:, 1].contiguous().view(-1)
            span_b_lin_loss = self.span_ce_loss(span_b_lin_scores, span_b_lin_targets)
            ## end of spans
            y_span_pre_e = y_rep
            for hw in self.span_e_proj:
                y_span_pre_e = hw(y_span_pre_e)
            span_e_scores = (x_reps[:, None, :, :] * y_span_pre_e[:, :, None, :]).sum(dim=-1)
            span_e_scores = (
                span_e_scores + (1 - y_mask_target.type_as(span_e_scores))[:, :, None] * 1e9
            )
            span_e_lin_scores = span_e_scores.view(-1, span_e_scores.shape[-1])
            span_e_lin_targets = y[:, 1:, 2].contiguous().view(-1)
            span_e_lin_loss = self.span_ce_loss(span_e_lin_scores, span_e_lin_targets)
            ## joint span prediction
            # TODO: predict full spans, enforce order
            # combine
            span_lin_loss = span_b_lin_loss + span_e_lin_loss
            span_loss = span_lin_loss.sum() / (y[:, :, 1] >= 0).sum()
            tot_loss = (1 - self.span_loss_lb) * lm_loss + self.span_loss_lb * span_loss
            # text span prediction
            # detach head
            if not is_eval:
                y_rep.retain_grad()
            self.text_span_start_hidden_z = y_rep.detach()
            self.text_span_end_hidden_z = y_rep.detach()
            self.text_span_start_hidden_z.requires_grad = True
            self.text_span_end_hidden_z.requires_grad = True
            self.text_span_start_hidden_z.retain_grad()
            self.text_span_end_hidden_z.retain_grad()
            # get predicted values
            self.text_span_start_out = self.text_span_start_head(self.text_span_start_hidden_z)
            text_span_start_scores = (
                x_reps[:, None, :, :] * self.text_span_start_out[:, :, None, :]
            ).sum(dim=-1)
            text_span_start_lin_scores = text_span_start_scores.view(
                -1, text_span_start_scores.shape[-1]
            )
            text_span_start_targets = y[:, 1:, 3].contiguous().view(-1)
            text_span_start_lin_loss = self.text_span_loss(
                text_span_start_lin_scores, text_span_start_targets
            )
            text_span_start_loss = text_span_start_lin_loss.sum() / (y[:, :, 3] >= 0).sum()
            # text span end prediction
            text_span_end_out = self.text_span_end_head(self.text_span_end_hidden_z)
            text_span_end_scores = (x_reps[:, None, :, :] * text_span_end_out[:, :, None, :]).sum(
                dim=-1
            )
            text_span_end_lin_scores = text_span_end_scores.view(
                -1, text_span_end_scores.shape[-1]
            )
            text_span_end_targets = y[:, 1:, 4].contiguous().view(-1)
            text_span_end_lin_loss = self.text_span_loss(
                text_span_end_lin_scores, text_span_end_targets
            )
            text_span_end_loss = text_span_end_lin_loss.sum() / (y[:, :, 4] >= 0).sum()
            text_span_lin_loss = text_span_start_loss + text_span_end_loss
            text_span_loss = text_span_lin_loss.sum() / (y[:, :, 3] >= 0).sum()
            res = {
                "lm_scores": lm_scores,
                "span_b_scores": span_b_scores,
                "span_e_scores": span_e_scores,
                "loss": tot_loss,
                "text_span_start_scores": text_span_start_scores,
                "text_span_end_scores": text_span_end_scores,
                "text_span_loss": text_span_loss,
            }
        return res


class EncoderDecoderWithLoss(nn.Module):
    """Model architecture that combines DecoderWithLoss with pre-trained BERT encoder.

    Attributes:
        encoder: Pre-trained BERT encoder
        decoder: Transformer decoder, see DecoderWithLoss
        train_encoder: whether to finetune the encoder model

    Args:
        encoder: Pre-trained BERT encoder
        decoder: Transformer decoder, see DecoderWithLoss
        args: Parsed command line args from running agent,
            eg. ``train_encoder`` specifies whether to train the encoder


    """

    def __init__(self, encoder, decoder, args):
        super(EncoderDecoderWithLoss, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_encoder = args.train_encoder

    def forward(self, x, x_mask, y, y_mask, x_reps=None, is_eval=False):
        if x_reps is None:
            model = self.encoder(input_ids=x, attention_mask=x_mask)
            x_reps = model[0]
        if not self.train_encoder:
            x_reps = x_reps.detach()
        # else:
        #     x_reps.retain_grad()
        outputs = self.decoder(y, y_mask, x_reps, x_mask, is_eval)
        return outputs


def predict_tree(txt, model, tokenizer, dataset, ban_noop=False, noop_threshold=0.0):
    """DEPRECATED: tree prediction. Use beam search."""
    model_device = model.decoder.lm_head.predictions.decoder.weight.device
    # prepare batch
    text, idx_maps = tokenize_mapidx(txt, tokenizer)
    tree = [("<S>", -1, -1)]
    text_idx_ls = [dataset.tokenizer._convert_token_to_id(w) for w in text.split()]
    tree_idx_ls = [[dataset.tree_idxs[w], bi, ei] for w, bi, ei in tree]
    pre_batch = [(text_idx_ls, tree_idx_ls, (text, txt, {}))]
    batch = caip_collate(pre_batch, tokenizer)
    batch = [t.to(model_device) for t in batch[:4]]
    x, x_mask, y, y_mask = batch
    y = y[:, :, 0]
    x_reps = model.encoder(input_ids=x, attention_mask=x_mask)[0].detach()
    res = [("<S>", -1, -1)]
    next_id = -1
    noop_predicted = False
    for i in range(100):
        if i > 0:
            y = torch.cat([y, torch.LongTensor([[next_id]]).to(model_device)], dim=1)
            y_mask = torch.cat(
                [y_mask, torch.LongTensor([1]).unsqueeze(dim=0).to(model_device)], dim=1
            )
        outputs = model.decoder.step(y, y_mask, x_reps, x_mask)
        # next word
        lm_scores = outputs["lm_scores"]
        s_lm_scores, s_lm_ids = lm_scores[0, -1].sort(dim=-1, descending=True)
        next_id = s_lm_ids[0].item()
        if "NOOP" in dataset.tree_voc[next_id]:
            if ban_noop or s_lm_scores[0].item() < noop_threshold:
                next_id = s_lm_ids[1].item()
                noop_predicted = True
                print("---- replacing NOOP with", dataset.tree_voc[next_id])
        next_w = dataset.tree_voc[next_id]
        # predicted span
        span_b_scores = outputs["span_b_scores"]
        span_e_scores = outputs["span_e_scores"]
        _, s_sb_ids = span_b_scores[0, -1].sort(dim=-1, descending=True)
        _, s_se_ids = span_e_scores[0, -1].sort(dim=-1, descending=True)
        b_id = s_sb_ids[0].item()
        e_id = s_se_ids[0].item()
        res += [(next_w, b_id, e_id)]
        if next_w == "</S>":
            break
    # only keep span predictions for span nodes, then map back to tree
    res = [(w, b, e) if w.startswith("BE:") else (w, -1, -1) for w, b, e in res]
    idx_rev_map = [(0, 0)] * len(text.split())
    for line_id, idx_map in enumerate(idx_maps):
        for pre_id, (a, b) in enumerate(idx_map):
            idx_rev_map[a] = (line_id, pre_id)
            idx_rev_map[b] = (line_id, pre_id)
    idx_rev_map[-1] = idx_rev_map[-2]
    res_tree, _ = seq_to_tree(dataset.full_tree, res[1:-1], idx_rev_map)
    return (res_tree, noop_predicted, (text, res))


def beam_search(txt, model, tokenizer, dataset, beam_size=5, well_formed_pen=1e2):
    """Beam search decoding.

    Note: Only uses node prediction scores, not the span scores.

    Args:
        txt (str): chat input
        model: model class with pretrained model
        tokenizer: pretrained tokenizer
        beam_size (int): Number of branches to keep in beam search
        well_formed_pen (float): penalization for poorly formed trees

    Returns:
        logical form (dict)

    """
    model_device = model.decoder.lm_head.predictions.decoder.weight.device
    # prepare batch
    text, idx_maps = tokenize_mapidx(txt, tokenizer)
    idx_rev_map = [(0, 0)] * len(text.split())
    for line_id, idx_map in enumerate(idx_maps):
        for pre_id, (a, b) in enumerate(idx_map):
            idx_rev_map[a] = (line_id, pre_id)
            idx_rev_map[b] = (line_id, pre_id)
    idx_rev_map[-1] = idx_rev_map[-2]
    tree = [("<S>", -1, -1, -1, -1)]
    text_idx_ls = [dataset.tokenizer._convert_token_to_id(w) for w in text.split()]
    tree_idx_ls = [
        [dataset.tree_idxs[w], bi, ei, text_span_bi, text_span_ei]
        for w, bi, ei, text_span_bi, text_span_ei in tree
    ]
    pre_batch = [(text_idx_ls, tree_idx_ls, (text, txt, {}))]
    batch = caip_collate(pre_batch, tokenizer)
    batch = [t.to(model_device) for t in batch[:4]]
    x, x_mask, y, y_mask = batch
    x_reps = model.encoder(input_ids=x, attention_mask=x_mask)[0].detach()
    x_mask = x_mask.expand(beam_size, -1)
    x_reps = x_reps.expand(beam_size, -1, -1)
    # start decoding
    y = torch.LongTensor([[dataset.tree_idxs["<S>"]] for _ in range(beam_size)]).to(
        model_device
    )  # B x 1
    beam_scores = torch.Tensor([-1e9 for _ in range(beam_size)]).to(model_device)  # B
    beam_scores[0] = 0
    beam_seqs = [[("<S>", -1, -1, -1, -1)] for _ in range(beam_size)]
    finished = [False for _ in range(beam_size)]
    pad_scores = torch.Tensor([-1e9] * len(dataset.tree_voc)).to(model_device)
    pad_scores[dataset.tree_idxs["[PAD]"]] = 0
    for i in range(100):
        outputs = model.decoder.step(y, y_mask, x_reps, x_mask)
        # next word
        lm_scores = outputs["lm_scores"][:, -1, :]  # B x V
        for i, fshed in enumerate(finished):
            if fshed:
                lm_scores[i] = pad_scores
        beam_lm_scores = lm_scores + beam_scores[:, None]  # B x V
        beam_lm_lin = beam_lm_scores.view(-1)
        s_scores, s_ids = beam_lm_lin.sort(dim=-1, descending=True)
        s_beam_ids = s_ids // beam_lm_scores.shape[-1]
        s_word_ids = s_ids % beam_lm_scores.shape[-1]
        # re-order and add next token
        beam_scores = s_scores[:beam_size]
        n_beam_ids = s_beam_ids[:beam_size]
        n_word_ids = s_word_ids[:beam_size]
        n_words = [dataset.tree_voc[nw_id.item()] for nw_id in n_word_ids]
        y = torch.cat([y[n_beam_ids], n_word_ids[:, None]], dim=1)
        # find out which of the beams are finished
        pre_finished = [finished[b_id.item()] for b_id in n_beam_ids]
        new_finished = [w_id.item() == dataset.tree_idxs["</S>"] for w_id in n_word_ids]
        finished = [p or n for p, n in zip(pre_finished, new_finished)]
        n_mask = 1 - torch.Tensor(finished).type_as(y_mask)
        y_mask = torch.cat([y_mask[n_beam_ids], n_mask[:, None]], dim=1)
        # predicted span
        span_b_scores = outputs["span_b_scores"][:, -1, :][n_beam_ids]  # B x T
        span_e_scores = outputs["span_e_scores"][:, -1, :][n_beam_ids]  # B x T
        span_be_scores = span_b_scores[:, :, None] + span_e_scores[:, None, :]
        invalid_scores = torch.tril(torch.ones(span_be_scores.shape), diagonal=-1) * -1e9
        span_be_scores += invalid_scores.type_as(span_be_scores)
        span_be_lin = span_be_scores.view(span_be_scores.shape[0], -1)
        _, s_sbe_ids = span_be_lin.sort(dim=-1, descending=True)
        s_sb_ids = s_sbe_ids[:, 0] // span_b_scores.shape[-1]
        s_se_ids = s_sbe_ids[:, 0] % span_b_scores.shape[-1]
        beam_b_ids = [bb_id.item() for bb_id in s_sb_ids]
        beam_e_ids = [be_id.item() for be_id in s_se_ids]

        # predict text spans
        text_span_start_scores = outputs["text_span_start_scores"][:, -1, :][n_beam_ids]  # B x T
        text_span_end_scores = outputs["text_span_end_scores"][:, -1, :][n_beam_ids]  # B x T
        text_span_scores = text_span_start_scores[:, :, None] + text_span_end_scores[:, None, :]
        invalid_text_span_scores = (
            torch.tril(torch.ones(text_span_scores.shape), diagonal=-1) * -1e9
        )
        text_span_scores += invalid_text_span_scores.type_as(text_span_scores)
        text_span_lin_scores = text_span_scores.view(text_span_scores.shape[0], -1)
        _, text_span_ids = text_span_lin_scores.sort(dim=-1, descending=True)
        text_span_start_ids = text_span_ids[:, 0] // text_span_start_scores.shape[-1]
        text_span_end_ids = text_span_ids[:, 0] % text_span_start_scores.shape[-1]
        text_span_beam_start_ids = [bb_id.item() for bb_id in text_span_start_ids]
        text_span_beam_end_ids = [be_id.item() for be_id in text_span_end_ids]

        # update beam_seq
        beam_seqs = [
            beam_seqs[n_beam_ids[i].item()]
            + [
                (
                    n_words[i],
                    beam_b_ids[i],
                    beam_e_ids[i],
                    text_span_beam_start_ids[i],
                    text_span_beam_end_ids[i],
                )
            ]
            for i in range(beam_size)
        ]
        # penalize poorly formed trees
        for i, seq in enumerate(beam_seqs):
            if seq[-1][0] == "</S>":
                _, well_formed = select_spans(seq)
                if not well_formed:
                    beam_scores[i] -= well_formed_pen
        # check whether all beams have reached EOS
        if all(finished):
            break
    # only keep span predictions for span nodes, then map back to tree
    beam_seqs = [
        [
            (w, b, e, -1, -1)
            if w.startswith("BE:")
            else (w, -1, -1, text_span_start, text_span_end)
            for w, b, e, text_span_start, text_span_end in res
            if w != "[PAD]"
        ]
        for res in beam_seqs
    ]
    beam_seqs = [
        [
            (w, -1, -1, text_span_start, text_span_end)
            if w.startswith("TBE:")
            else (w, b, e, -1, -1)
            for w, b, e, text_span_start, text_span_end in res
            if w != "[PAD]"
        ]
        for res in beam_seqs
    ]
    # delinearize predicted sequences into tree
    beam_trees = [seq_to_tree(dataset.full_tree, res[1:-1], idx_rev_map)[0] for res in beam_seqs]
    pre_res = [
        (tree, score.item(), seq) for tree, score, seq in zip(beam_trees, beam_scores, beam_seqs)
    ]
    # sort one last time to have well-formed trees on top
    res = sorted(pre_res, key=lambda x: x[1], reverse=True)
    return res


def compute_accuracy(outputs, y):
    """Util function for validation.

    Args:
        outputs: targets
        y: predicted sequence

    Returns:
        Either a tuple of LM and span accuracies or just the language modeling accuracy.

    """
    if len(y.shape) == 2:
        lm_targets = y
    else:
        lm_targets = y[:, 1:, 0]

    lm_preds = outputs["lm_scores"].max(dim=-1)[1]
    lm_acc = ((lm_preds == lm_targets) * (lm_targets > 6)).sum(dim=1) == (lm_targets > 6).sum(
        dim=1
    )
    if "span_b_scores" in outputs:
        sb_targets = y[:, 1:, 1]
        sb_preds = outputs["span_b_scores"].max(dim=-1)[1]
        sb_acc = ((sb_preds == sb_targets) * (sb_targets >= 0)).sum(dim=1) == (
            sb_targets >= 0
        ).sum(dim=1)
        se_targets = y[:, 1:, 2]
        se_preds = outputs["span_e_scores"].max(dim=-1)[1]
        se_acc = ((se_preds == se_targets) * (se_targets >= 0)).sum(dim=1) == (
            se_targets >= 0
        ).sum(dim=1)
        sp_acc = sb_acc * se_acc
        full_acc = lm_acc * sp_acc
        if "text_span_start_scores" in outputs:
            text_span_b_targets = y[:, 1:, 3]
            text_span_e_targets = y[:, 1:, 4]
            text_span_b_pred = outputs["text_span_start_scores"].max(dim=-1)[1]
            text_span_e_pred = outputs["text_span_end_scores"].max(dim=-1)[1]
            text_span_b_acc = (
                (text_span_b_pred == text_span_b_targets) * (text_span_b_targets >= 0)
            ).sum(dim=1) == (text_span_b_targets >= 0).sum(dim=1)
            text_span_e_acc = (
                (text_span_e_pred == text_span_e_targets) * (text_span_e_targets >= 0)
            ).sum(dim=1) == (text_span_e_targets >= 0).sum(dim=1)
            text_span_acc = text_span_b_acc * text_span_e_acc
            return (lm_acc, sp_acc, text_span_acc, full_acc)
        else:
            return (lm_acc, sp_acc, full_acc)
    else:
        return lm_acc


class OptimWarmupEncoderDecoder(object):
    """Custom wrapper for Adam optimizer, handles lr warmup and smaller lr for encoder fine-tuning."""

    def __init__(self, model, args):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = {
            "encoder": args.encoder_learning_rate,
            "decoder": args.decoder_learning_rate,
            "text_span_decoder": args.decoder_learning_rate,
        }
        self.warmup_steps = {
            "encoder": args.encoder_warmup_steps,
            "decoder": args.decoder_warmup_steps,
            "text_span_decoder": args.decoder_warmup_steps,
        }
        if args.optimizer == "adam":
            self.optimizers = {
                "encoder": Adam(model.encoder.parameters(), lr=self.lr["encoder"]),
                "decoder": Adam(model.decoder.parameters(), lr=self.lr["decoder"]),
                "text_span_decoder": Adam(
                    model.decoder.parameters(), lr=self.lr["text_span_decoder"]
                ),
            }
        elif args.optimizer == "adagrad":
            self.optimizers = {
                "encoder": Adagrad(model.encoder.parameters(), lr=self.lr["encoder"]),
                "decoder": Adagrad(model.decoder.parameters(), lr=self.lr["decoder"]),
                "text_span_decoder": Adam(
                    model.decoder.parameters(), lr=self.lr["text_span_decoder"]
                ),
            }
        else:
            raise NotImplementedError

        self._step = 0

    def _update_rate(self, stack):
        return self.lr[stack] * min(
            (self._step / self.warmup_steps[stack]), (self._step / self.warmup_steps[stack]) ** 0.5
        )

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()


# --------------------------
# Label smoothing loss
# --------------------------
class LabelSmoothingLoss(nn.Module):
    """With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.

    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-1):
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.voc_size = tgt_vocab_size
        if ignore_index >= 0:
            self.smoothing = label_smoothing / (tgt_vocab_size - 2)
        else:
            self.smoothing = label_smoothing / (tgt_vocab_size - 1)
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """Forward method for LSL.

        Args:
            output (FloatTensor): batch_size x n_classes
            target (LongTensor): batch_size

        """
        with torch.no_grad():
            s_target = torch.zeros_like(output)
            s_target.fill_(self.smoothing)
            if self.ignore_index >= 0:
                s_target[:, self.ignore_index] = 0
            t_cap = target.masked_fill(target == self.ignore_index, 0)
            s_target.scatter_(1, t_cap.unsqueeze(1), self.confidence)

        kl_div = F.kl_div(output.log_softmax(dim=-1), s_target, reduction="none")
        kl_mask = (target != self.ignore_index).type_as(kl_div).unsqueeze(1)
        return (kl_div * kl_mask).sum(dim=-1)
