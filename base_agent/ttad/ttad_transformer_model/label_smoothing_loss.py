import torch
import torch.nn as nn

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
