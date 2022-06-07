#!/usr/bin/env python
import os

import pickle 
import numpy as np
from PIL import Image
from eyehandcal.utils import detect_corners

with open('caldata_jay.pkl', 'rb') as f:
    data = pickle.load(f)    

caldata = 'caldata_imgs'
os.makedirs(caldata, exist_ok=True)
corner_data = detect_corners(data, target_idx=0)
for i, d in enumerate(data):
    for j,img in enumerate(d['imgs']):
        img_pil = Image.fromarray(img.astype(np.uint8), mode='RGB')
        fname = f'{caldata}/shot_{i}_cam_{j}.jpg'
        img_pil.save(fname)
        print(fname, corner_data[i]['corners'][j])

