#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from eyehandcal.utils import detect_corners
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('caldata')
parser.add_argument('outdata')
args = parser.parse_args()

with open(args.caldata, 'rb') as f:
    data = pickle.load(f)

if isinstance(data[0]['intrinsics'], OrderedDict):
    print('intrinsics already in new data format')
    raise SystemExit

new_intrinsics = OrderedDict()
for i, intrinsics_dict in enumerate(data[0]['intrinsics']):

    intrinsics_dict['model'] = intrinsics_dict['model'].name
    new_intrinsics[f'sample_serial_{i}'] = intrinsics_dict
        
data[0]['intrinsics'] = new_intrinsics

print(data[0]['intrinsics'])
with open(args.outdata, 'wb') as f:
    pickle.dump(data, f)
