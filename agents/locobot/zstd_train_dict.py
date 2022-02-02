import pyzstd as zstd
import numpy as np
import random
import pathlib
import pickle

random.seed(0)

def train_data(path = '.', dataset_size = 1000):
    train_files = list(pathlib.Path(path).glob('**/depth/*.npy'))
    random.shuffle(train_files)
    for file_ in train_files[0:dataset_size]:
        data = np.load(file_)
        yield data.tobytes()


train_iter = train_data()
dic = zstd.train_dict(train_iter, 100*1024)
with open("zstd_dict_depth.pkl", "wb") as f:
    pickle.dump(dic.dict_content, f)

# train_iter = train_data()
# final_dic = zstd.finalize_dict(dic, train_iter, 100*1024, 0)

