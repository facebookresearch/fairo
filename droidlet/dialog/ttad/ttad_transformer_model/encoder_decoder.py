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

    def forward(self, x, x_mask, y, y_mask, x_reps=None, is_eval=False):
        if x_reps is None:
            model = self.encoder(input_ids=x, attention_mask=x_mask)
            x_reps = model[0]
        if not self.train_encoder:
            x_reps = x_reps.detach()
        outputs = self.decoder(y, y_mask, x_reps, x_mask, is_eval)
        return outputs
