"""
Copyright (c) Facebook, Inc. and its affiliates.

Utils for processing model log.
"""
import math


def read_model_log_to_list(fname: str):
    """
    Read model log and export to a dictionary including:
        - Epoch loss and accuracy
        - Text span loss and accuracy
    """
    f = open(fname)
    content = f.read()
    f.close()

    lines = content.split("\n")

    def get_loss_acc(line: str):
        words = line.split(" ")
        loss, acc = None, None
        for i in range(0, len(words)):
            if i > 0 and words[i - 1] == "Loss:":
                loss = float(words[i])
                loss = None if math.isnan(loss) else loss
            elif i > 0 and words[i - 1] == "Accuracy:":
                acc = float(words[i])
                acc = None if acc is math.isnan(acc) else acc
        return loss, acc

    epoch_loss_acc_list = []
    text_span_loss_acc_list = []
    for line in lines:
        if "epoch" in line:
            loss, acc = get_loss_acc(line)
            epoch_loss_acc_list.append({"loss": loss, "acc": acc})
        elif " text span Loss: " in line:
            loss, acc = get_loss_acc(line)
            text_span_loss_acc_list.append({"loss": loss, "acc": acc})

    return epoch_loss_acc_list, text_span_loss_acc_list
