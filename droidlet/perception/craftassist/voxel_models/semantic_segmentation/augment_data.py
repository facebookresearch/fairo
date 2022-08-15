import copy
import pickle
import numpy as np
import os
import random

seed = 123

random.seed(seed)
np.random.seed(seed)



original_data_path = "/checkpoint/yuxuans/datasets/inst_seg/turk_annotated_data_0/1659755341_clean_modeldata.pkl"
split_ratio = 0.8
output_dir = f"/checkpoint/yuxuans/datasets/inst_seg/aug_turk_data_seed{seed}/"

def augment_data(data):
    new_data = []
    S, L, tags = data

    # swap x, z axis
    NS = S.swapaxes(0, 2)
    NL = L.swapaxes(0, 2)
    d = copy.deepcopy([NS, NL, tags])
    new_data.append(d)

    # flip axis along x, y, z separately
    for axis in range(3):
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.flip(NS, axis)
        np.flip(NL, axis)
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)
    
    for shift_d in [2]:
        # positive shift x
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((shift_d, 0), (0, 0), (0, 0)), mode='constant')[: -shift_d, :, :]
        np.pad(NL, ((shift_d, 0), (0, 0), (0, 0)), mode='constant')[: -shift_d, :, :]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)
        # negative shift x
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((0, shift_d), (0, 0), (0, 0)), mode='constant')[shift_d:, :, :]
        np.pad(NL, ((0, shift_d), (0, 0), (0, 0)), mode='constant')[shift_d:, :, :]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)

        # positive shift y
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((0, 0), (shift_d, 0), (0, 0)), mode='constant')[:, : -shift_d, :]
        np.pad(NL, ((0, 0), (shift_d, 0), (0, 0)), mode='constant')[:, : -shift_d, :]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)
        # negative shift y
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((0, 0), (0, shift_d), (0, 0)), mode='constant')[:, shift_d: , :]
        np.pad(NL, ((0, 0), (0, shift_d), (0, 0)), mode='constant')[:, shift_d: , :]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)

        # positive shift z
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((0, 0), (0, 0), (shift_d, 0)), mode='constant')[:, :, : -shift_d]
        np.pad(NL, ((0, 0), (0, 0), (shift_d, 0)), mode='constant')[:, :, : -shift_d]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)
        # negative shift z
        NS = copy.deepcopy(S)
        NL = copy.deepcopy(L)
        np.pad(NS, ((0, 0), (0, 0), (0, shift_d)), mode='constant')[:, :, shift_d: ]
        np.pad(NL, ((0, 0), (0, 0), (0, shift_d)), mode='constant')[:, :, shift_d: ]
        d = copy.deepcopy([NS, NL, tags])
        new_data.append(d)

    return new_data


def augment_dataset(dataset):
    new_data = []
    for data in dataset:
        augmented_data = augment_data(data)
        for d in augmented_data:
            new_data.append(d)
    return new_data

original_data = pickle.load(open(original_data_path, "rb"))
train_len = int(len(original_data) * split_ratio)
train_data = original_data[:train_len]
valid_data = original_data[train_len:]

augmented_train_data = augment_dataset(train_data)
augmented_valid_data = augment_dataset(valid_data)

random.shuffle(augmented_train_data)
random.shuffle(augmented_valid_data)

print(f"train data sz: {len(train_data)}, augmented train data sz: {len(augmented_train_data)}")
print(f"valid data sz: {len(valid_data)}, augmented valid data sz: {len(augmented_valid_data)}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dir not exists, creating it: {output_dir}")

with open(f"{output_dir}training_data.pkl", "wb") as f:
    pickle.dump(augmented_train_data, f)
with open(f"{output_dir}validation_data.pkl", "wb") as f:
    pickle.dump(augmented_valid_data, f)