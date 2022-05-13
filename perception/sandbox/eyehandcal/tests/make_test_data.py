#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from eyehandcal.utils import detect_corners
import os
import cv2
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('caldata')
parser.add_argument('outdata')
args = parser.parse_args()

with open(args.caldata, 'rb') as f:
    data = pickle.load(f)

random.shuffle(data)
data=data[:20]


detect_corners(data)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),50]

for i, d in enumerate(data):
    compressed_imgs = []
    for img in d['imgs']:
        status, img_jpeg_encoded = cv2.imencode('.jpg', img, encode_param)
        compressed_imgs.append(img_jpeg_encoded)
    d['imgs_jpeg_encoded'] = compressed_imgs
    del d['imgs']

with open(args.outdata, 'wb') as f:
    pickle.dump(data, f)
