import torch
import torch.nn as nn
import logging
from .modeling_bert import BertModel, BertOnlyMLMHead
from .tokenization_utils import fixed_span_values_voc


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
        logging.debug("initializing decoder with params {}".format(args))
        self.bert = BertModel(config)
        self.lm_head = BertOnlyMLMHead(config)
        self.fixed_span_head = nn.Linear(config.hidden_size, len(fixed_span_values_voc))
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
        self.fixed_span_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
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
            labels=y,
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
        fixed_value_scores = self.fixed_span_head(y_rep)
        res = {
            "lm_scores": torch.log_softmax(lm_scores, dim=-1).detach(),
            "span_b_scores": torch.log_softmax(span_b_scores, dim=-1).detach(),
            "span_e_scores": torch.log_softmax(span_e_scores, dim=-1).detach(),
            "text_span_start_scores": torch.log_softmax(text_span_start_scores, dim=-1).detach(),
            "text_span_end_scores": torch.log_softmax(text_span_end_scores, dim=-1).detach(),
            "fixed_value_scores": torch.log_softmax(fixed_value_scores, dim=-1).detach(),
        }
        return res

    def forward(self, labels, y, y_mask, x_reps, x_mask, is_eval=False):
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
                labels=y,
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
            # fixed span value output head
            self.fixed_span_hidden_z = y_rep.detach()
            self.fixed_span_hidden_z.requires_grad = True
            self.fixed_span_hidden_z.retain_grad()
            fixed_span_scores = self.fixed_span_head(self.fixed_span_hidden_z)
            fixed_span_lin_scores = fixed_span_scores.view(-1, fixed_span_scores.shape[-1])
            fixed_span_lin_targets = y[:, 1:, 5].contiguous().view(-1)
            fixed_span_lin_loss = self.fixed_span_loss(
                fixed_span_lin_scores, fixed_span_lin_targets
            )
            fixed_span_loss = fixed_span_lin_loss.sum() / (y[:, :, 5] >= 0).sum()
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
                "fixed_span_loss": fixed_span_loss,
            }
        return res
