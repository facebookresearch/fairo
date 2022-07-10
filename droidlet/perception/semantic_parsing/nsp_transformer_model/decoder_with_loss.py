import torch
import torch.nn as nn
import logging
from droidlet.perception.semantic_parsing.nsp_transformer_model.modeling_bert import (
    BertModel,
    BertOnlyMLMHead,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.tokenization_utils import (
    fixed_span_values_voc,
)
from droidlet.perception.semantic_parsing.nsp_transformer_model.label_smoothing_loss import (
    LabelSmoothingLoss,
)


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
        text_span_loss: Cross Entropy loss for text spans

    """

    def __init__(self, config, args, tokenizer):
        super(DecoderWithLoss, self).__init__()
        # model components
        logging.debug("initializing decoder with params {}".format(args))
        self.bert = BertModel(config)
        # language modeling head
        self.lm_head = BertOnlyMLMHead(config)
        # loss functions
        if args.node_label_smoothing > 0:
            self.lm_ce_loss = LabelSmoothingLoss(
                args.node_label_smoothing, config.vocab_size, ignore_index=tokenizer.pad_token_id
            )
        else:
            self.lm_ce_loss = torch.nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id, reduction="none"
            )
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
        lm_scores = self.lm_head(y_rep)

        res = {
            "lm_scores": torch.log_softmax(lm_scores, dim=-1).detach(),
        }
        return res

    def forward(self, labels, y, y_mask, x_reps, x_mask, is_eval=False):
        """Same as step, except with loss. Set is_eval=True for validation.

        labels: B x y_len x num_heads
        y: B x y_len x num_heads
        y_mask: B x y_len x num_heads
        x_reps: B x x_len x H
        x_mask: B x x_len x H

        output heads are (lm, span_start, span_end, text_span_start, text_span_end, fixed_value)
        For shape of outputs, see forward() in encoder_decoder.py
        """
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
                input_ids=y[:, :-1],
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
            lm_lin_targets = y[:, 1:].contiguous().view(-1)
            lm_lin_loss = self.lm_ce_loss(lm_lin_scores, lm_lin_targets)
            lm_lin_mask = y_mask_target.view(-1)
            lm_loss = lm_lin_loss.sum() / lm_lin_mask.sum()
            tot_loss = lm_loss

            res = {
                "lm_scores": lm_scores,
                "loss": tot_loss,
            }
        return res
