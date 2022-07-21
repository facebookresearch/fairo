import pandas as pd

def log_to_df(fname: str):
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
            elif i > 0 and words[i - 1] == "Accuracy:":
                acc = float(words[i])
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

    return pd.DataFrame(epoch_loss_acc_list), pd.DataFrame(text_span_loss_acc_list)
    
