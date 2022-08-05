"""
Copyright (c) Facebook, Inc. and its affiliates.

Utils for processing model log.
"""
import math


def read_model_log_to_list(fname: str):
    """
    Read model log and export to a dictionary including:
        - Loss changes over epoch increases of training / validation
        - Accuracy changes over epoch increases of training / validation
    """

    def get_loss_acc(line: str):
        words = line.split(" ")
        loss, acc = None, None
        for i in range(0, len(words)):
            if i > 0 and (words[i - 1] == "Loss:" or words[i - 1] == "L:"):
                loss = float(words[i])
                loss = None if math.isnan(loss) else loss
            elif i > 0 and (words[i - 1] == "Accuracy:" or words[i - 1] == "A:"):
                acc = float(words[i])
                acc = None if acc is math.isnan(acc) else acc
        return loss, acc

    loss_list = []
    acc_list = []
    t_key, v_key = "training", "validation"

    f = open(fname)
    for line in f:
        if "Epoch: " in line:
            loss_list.append({})
            acc_list.append({})
        elif "L: " in line:
            # training
            loss, acc = get_loss_acc(line)
            loss_list[-1][t_key] = loss
            acc_list[-1][t_key] = acc
        elif "valid: " in line:
            # validation
            loss, acc = get_loss_acc(line)
            loss_list[-1][v_key] = loss
            acc_list[-1][v_key] = acc
    f.close()
    return loss_list, acc_list
