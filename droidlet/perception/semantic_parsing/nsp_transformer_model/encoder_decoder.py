from torch import nonzero
import torch.nn as nn


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

        if args.pretrained_encoder_name == "bert-large-uncased" and args.decoder_config_name == "bert-base-uncased":
            self.proj_encoder_decoder = nn.Linear(1024, 768)
        else:
            self.proj_encoder_decoder = None

    def forward(self, x, x_mask, y, y_mask, x_reps=None, is_eval=False):
        """
        Shapes of inputs to forward:

        x: [B, x_len]
        x_mask: [B, x_len]
        y: [B, y_len, num_heads]
        y_mask: [B, y_len]

        Shapes of outputs:

        outputs: a dictionary of output scores.
        -- lm_scores: [B, y_len, V] # Note - excludes first token
        -- span_b_scores: [B, y_len, span_range]
        -- span_e_scores: [B, y_len, span_range]
        -- loss: [float]
        -- text_span_start_scores: [B, y_len, span_range]
        -- text_span_end_scores: [B, y_len, span_range]
        -- text_span_loss: [float]
        -- fixed_span_loss: [float]
        """
        if x_reps is None:
            model = self.encoder(input_ids=x, attention_mask=x_mask)
            x_reps = model[0]
        if not self.train_encoder:
            x_reps = x_reps.detach()
        
        if self.proj_encoder_decoder:
            x_reps = self.proj_encoder_decoder(x_reps)

        outputs = self.decoder(y, y, y_mask, x_reps, x_mask, is_eval)
        return outputs
