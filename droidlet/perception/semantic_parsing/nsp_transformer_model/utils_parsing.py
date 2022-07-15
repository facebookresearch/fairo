"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from black import NothingChanged
import torch
import json
from .utils_caip import select_spans, seq_to_tree, tokenize_mapidx, caip_collate
from .tokenization_utils import fixed_span_values_voc


def beam_search(txt, model, tokenizer, dataset, beam_size=5, well_formed_pen=1e2):
    """
    Beam search decoding.
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
    tree = [("<S>", -1, -1, -1)]
    text_idx_ls = dataset.tokenizer.convert_tokens_to_ids(text.split())
    tree_idx_ls = [
        [dataset.tree_idxs[w], text_span_bi, text_span_ei, fixed_val]
        for w, text_span_bi, text_span_ei, fixed_val in tree
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
    beam_seqs = [[("<S>", -1, -1, -1)] for _ in range(beam_size)]
    finished = [False for _ in range(beam_size)]
    fixed_value_vocab_size = len(fixed_span_values_voc)
    pad_scores = torch.Tensor([-1e9] * (len(dataset.tree_voc) - fixed_value_vocab_size)).to(
        model_device
    )
    pad_scores[dataset.tree_idxs["[PAD]"]] = 0
    for i in range(100):
        outputs = model.decoder.step(y, y_mask, x_reps, x_mask)
        # next word, grab the final token
        lm_scores = outputs["lm_scores"][:, -1, :]  # B x V
        for i, fshed in enumerate(finished):
            if fshed:
                # set predictions to padding tokens
                lm_scores[i] = pad_scores
        beam_lm_scores = lm_scores + beam_scores[:, None]  # B x V
        beam_lm_lin = beam_lm_scores.view(-1)
        # get the highest probability tokens
        s_scores, s_ids = beam_lm_lin.sort(dim=-1, descending=True)
        s_beam_ids = s_ids // beam_lm_scores.shape[-1]
        s_word_ids = s_ids % beam_lm_scores.shape[-1]
        # re-order and add next token
        beam_scores = s_scores[:beam_size]
        n_beam_ids = s_beam_ids[:beam_size]
        n_word_ids = s_word_ids[:beam_size]
        # convert tokens to words
        n_words = [dataset.tree_voc[nw_id.item()] for nw_id in n_word_ids]
        y = torch.cat([y[n_beam_ids], n_word_ids[:, None]], dim=1)
        # find out which of the beams are finished
        pre_finished = [finished[b_id.item()] for b_id in n_beam_ids]
        new_finished = [w_id.item() == dataset.tree_idxs["</S>"] for w_id in n_word_ids]
        finished = [p or n for p, n in zip(pre_finished, new_finished)]
        n_mask = 1 - torch.Tensor(finished).type_as(y_mask)
        y_mask = torch.cat([y_mask[n_beam_ids], n_mask[:, None]], dim=1)

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

        # predict fixed values
        fixed_value_scores = outputs["fixed_value_scores"][:, -1, :][n_beam_ids]  # B x T
        fixed_value_lin_scores = fixed_value_scores.view(-1)
        # get the highest probability tokens
        fixed_value_ranked_scores, fixed_value_ids = fixed_value_lin_scores.sort(
            dim=-1, descending=True
        )
        fixed_value_beam_ids = fixed_value_ids // fixed_value_scores.shape[-1]
        # map back to which word in sequence, since
        fixed_value_word_ids = fixed_value_ids % fixed_value_scores.shape[-1]
        # re-order and add next token
        fixed_value_beam_scores = fixed_value_ranked_scores[:beam_size]
        fixed_value_beam_ids = fixed_value_beam_ids[:beam_size]
        fixed_value_word_ids = fixed_value_word_ids[:beam_size]
        # convert tokens to words
        fixed_value_words = [fixed_span_values_voc[nw_id.item()] for nw_id in fixed_value_word_ids]

        # update beam_seq
        beam_seqs = [
            beam_seqs[n_beam_ids[i].item()]
            + [
                (
                    n_words[i],
                    text_span_beam_start_ids[i],
                    text_span_beam_end_ids[i],
                    fixed_value_words[i],
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
            (w, text_span_start, text_span_end, -1)
            if w.startswith("BE:")
            else (w, text_span_start, text_span_end, fixed_val)
            for w, text_span_start, text_span_end, fixed_val in res
            if w != "[PAD]"
        ]
        for res in beam_seqs
    ]
    beam_seqs = [
        [
            (w, text_span_start, text_span_end, -1)
            if w.startswith("TBE:")
            else (w, text_span_start, text_span_end, fixed_val)
            for w, text_span_start, text_span_end, fixed_val in res
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


def beam_search_simp(txt, model, tokenizer, dataset, beam_size=5, well_formed_pen=1e2):
    """
    Beam search decoding with only language modelling head.
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
    tree = ["[CLS]"]
    text_idx_ls = dataset.tokenizer.convert_tokens_to_ids(text.split())
    tree_idx_ls = dataset.tokenizer.convert_tokens_to_ids(tree)
    pre_batch = [(text_idx_ls, tree_idx_ls, (text, txt, {}))]
    batch = caip_collate(pre_batch, tokenizer)
    batch = [t.to(model_device) for t in batch[:4]]
    x, x_mask, y, y_mask = batch
    x_reps = model.encoder(input_ids=x, attention_mask=x_mask)[0].detach()
    x_mask = x_mask.expand(beam_size, -1)
    x_reps = x_reps.expand(beam_size, -1, -1)
    # start decoding
    y = torch.LongTensor(
        [[dataset.tokenizer.convert_tokens_to_ids("[CLS]")] for _ in range(beam_size)]
    ).to(
        model_device
    )  # B x 1
    beam_scores = torch.Tensor([-1e9 for _ in range(beam_size)]).to(model_device)  # B
    beam_scores[0] = 0
    beam_seqs = [["[CLS]"] for _ in range(beam_size)]
    finished = [False for _ in range(beam_size)]
    pad_scores = torch.Tensor([-1e9] * dataset.tokenizer.vocab_size).to(model_device)
    pad_scores[dataset.tokenizer.convert_tokens_to_ids("[PAD]")] = 0
    for i in range(512):
        outputs = model.decoder.step(y, y_mask, x_reps, x_mask)
        # next word, grab the final token
        lm_scores = outputs["lm_scores"][:, -1, :]  # B x V
        for i, fshed in enumerate(finished):
            if fshed:
                # set predictions to padding tokens
                lm_scores[i] = pad_scores
        beam_lm_scores = lm_scores + beam_scores[:, None]  # B x V
        beam_lm_lin = beam_lm_scores.view(-1)
        # get the highest probability tokens
        s_scores, s_ids = beam_lm_lin.sort(dim=-1, descending=True)
        s_beam_ids = s_ids // beam_lm_scores.shape[-1]
        s_word_ids = s_ids % beam_lm_scores.shape[-1]
        # re-order and add next token
        beam_scores = s_scores[:beam_size]
        n_beam_ids = s_beam_ids[:beam_size]
        n_word_ids = s_word_ids[:beam_size]
        # convert tokens to words
        n_words = [tokenizer.convert_ids_to_tokens(nw_id.item()) for nw_id in n_word_ids]
        y = torch.cat([y[n_beam_ids], n_word_ids[:, None]], dim=1)
        # find out which of the beams are finished
        pre_finished = [finished[b_id.item()] for b_id in n_beam_ids]
        new_finished = [
            w_id.item() == dataset.tokenizer.convert_tokens_to_ids("[SEP]") for w_id in n_word_ids
        ]
        finished = [p or n for p, n in zip(pre_finished, new_finished)]
        n_mask = 1 - torch.Tensor(finished).type_as(y_mask)
        y_mask = torch.cat([y_mask[n_beam_ids], n_mask[:, None]], dim=1)

        # update beam_seq
        beam_seqs = [beam_seqs[n_beam_ids[i].item()] + [n_words[i]] for i in range(beam_size)]
        # penalize poorly formed trees
        for i, seq in enumerate(beam_seqs):
            if seq[-1] == "[SEP]":
                well_formed = check_tree_well_formed(seq)
                if not well_formed:
                    beam_scores[i] -= well_formed_pen
        # check whether all beams have reached EOS
        if all(finished):
            break

    # map tokenized sequence of tree back to tree
    beam_trees = [detokenize_tree(res, dataset.tokenizer) for res in beam_seqs]
    pre_res = [
        (tree, score.item(), seq) for tree, score, seq in zip(beam_trees, beam_scores, beam_seqs)
    ]
    # sort one last time to have well-formed trees on top
    res = sorted(pre_res, key=lambda x: x[1], reverse=True)
    return res


def check_tree_well_formed(tok_tree):
    """
    Check if the syntactic context of tree is well predicted via pairing
    left and right curly and square brackets

    Args:
        tok_tree: predicted tree tokens

    Returns:
        well_formed: boolean, whether all left brackets are paired to
        right brackets
    """
    queue = []
    for tok in tok_tree:
        if tok == "{":
            queue.append(tok)
        elif tok == "}":
            if queue and queue[-1] == "{":
                queue.pop()
            else:
                return False
        elif tok == "[":
            queue.append(tok)
        elif tok == "]":
            if queue and queue[-1] == "[":
                queue.pop()
            else:
                return False

    return len(queue) == 0


def detokenize_tree(seq, tokenizer):
    """
    Transform tokenized sequence of tree back to tree

    Args:
        seq: list of tokens
        tokenizer: pretrained tokenizer

    Returns:
        tree: dictionaru
    """
    tok_tree = tokenizer.convert_tokens_to_string(seq)
    tok_tree = tok_tree.split()

    special_tokens = ["[", "]", "{", "}", ":", ",", "_", '"', '",', ".", "/"]
    tree = ""
    prev_token = None
    for token in tok_tree:
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        else:
            # deal with - symbol
            if token == "-":
                if prev_token in special_tokens:
                    tree += token
                else:
                    tree += " " + token
            elif token not in special_tokens and prev_token == "-":
                tree += token
            # add space between texts of span node
            elif token not in special_tokens and prev_token not in special_tokens:
                tree += " " + token
            else:
                tree += token
            prev_token = token

    return json.loads(tree)


def compute_accuracy(outputs, y):
    """Util function for validation.

    Args:
        outputs: targets
            A dictionary of output values from forward consisting of LM scores, span scores etc.
            -- lm_scores: [B, y_len, V] # Note - excludes first token
            -- span_b_scores: [B, y_len, span_range]
            -- span_e_scores: [B, y_len, span_range]
            -- loss: [float]
            -- text_span_start_scores: [B, y_len, span_range]
            -- text_span_end_scores: [B, y_len, span_range]
            -- text_span_loss: [float]
            -- fixed_span_loss: [float]
        y: predicted sequence

    Returns:
        Either a tuple of LM and span accuracies or just the language modeling accuracy.

    Shape of inputs:
    y: B x y_len x num_heads
    """
    if len(y.shape) == 2:
        lm_targets = y[:, 1:]
    else:
        lm_targets = y[:, 1:, 0]

    lm_preds = outputs["lm_scores"].max(dim=-1)[1]
    lm_acc = ((lm_preds == lm_targets) * (lm_targets > 101)).sum(dim=1) == (lm_targets > 101).sum(
        dim=1
    )
    full_acc = lm_acc

    if "text_span_start_scores" in outputs:
        text_span_b_targets = y[:, 1:, 1]
        text_span_e_targets = y[:, 1:, 2]
        text_span_b_pred = outputs["text_span_start_scores"].max(dim=-1)[1]
        text_span_e_pred = outputs["text_span_end_scores"].max(dim=-1)[1]
        text_span_b_acc = (
            (text_span_b_pred == text_span_b_targets) * (text_span_b_targets >= 0)
        ).sum(dim=1) == (text_span_b_targets >= 0).sum(dim=1)
        text_span_e_acc = (
            (text_span_e_pred == text_span_e_targets) * (text_span_e_targets >= 0)
        ).sum(dim=1) == (text_span_e_targets >= 0).sum(dim=1)
        text_span_acc = text_span_b_acc * text_span_e_acc
        return (lm_acc, text_span_acc, full_acc)
    else:
        return (lm_acc, full_acc)
