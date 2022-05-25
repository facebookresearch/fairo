#%%
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Read the weight CSV into epoch series
with open("weights.csv", "r") as f:
    weightslist = f.readlines()

X = list(range(len(weightslist)))
Ys = []
for weights in weightslist:
    Ys.append(weights.strip().split(","))

num_series = len(Ys[0])
series = [[] for x in range(num_series)]
for epoch in range(len(Ys)):
    for i in range(num_series):
        series[i].append(float(Ys[epoch][i]))

plt.xlabel("Epoch")
plt.ylabel("Module Weights Norm")
colortick = 0.8 / num_series
for i in range(len(series)):
    plt.plot(X, series[i], color=(0.05, 0.4, 0.05, (0.2 + (i*colortick))))
plt.ylim(0, 90)
plt.show()

# %%
# Read the gradient CSV into epoch series
with open("gradients.csv", "r") as f:
    gradslist = f.readlines()

X = list(range(len(gradslist)))
Ys = []
for grads in gradslist:
    Ys.append(grads.strip().split(","))

num_series = len(Ys[0])
series = [[] for x in range(num_series)]

for epoch in range(len(Ys)):
    for i in range(num_series):
        series[i].append(float(Ys[epoch][i]))

plt.xlabel("Epoch")
plt.ylabel("Module Gradients Norm")
colortick = 0.8 / num_series
for i in range(len(series)):
    plt.plot(X, series[i], color=(0.4, 0.05, 0.05, (0.2 + (i*colortick))))
plt.show()

# %%
# Plot training and valid accuracy

train_csv = pd.read_csv("training_outputs.csv", delimiter='|', header=0)
valid_csv = pd.read_csv("valid_outputs.csv", delimiter='|', header=0)

train_accs = []
for idx, row in train_csv.iterrows():
    if int(row["iteration"]) == 400:
        train_accs.append(row["accuracy"])

anno_valid_accs = []
clar_valid_accs = []
for idx, row in valid_csv.iterrows():
    if row["data_type"] == "annotated":
        anno_valid_accs.append(row["accuracy"])
    if row["data_type"] == "templated_clarification":
        clar_valid_accs.append(row["accuracy"])

Xt = list(range(len(train_accs)))
Xv = list(range(len(anno_valid_accs)))
Xc = list(range(len(clar_valid_accs)))

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(Xt, train_accs, color='g', label='Train')
plt.plot(Xv, anno_valid_accs, color='b', label='Annotated Valid')
plt.plot(Xc, clar_valid_accs, color='r', label='Clarification Valid')
plt.legend()
plt.show()


# %%
