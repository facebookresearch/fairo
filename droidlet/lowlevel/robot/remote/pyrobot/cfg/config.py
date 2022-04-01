# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN

_C = CN()

_C.HAS_BASE = True
_C.HAS_CAMERA = True
_C.HAS_COMMON = False
_C.CAMERA = CN()
_C.BASE = CN()
_C.COMMON = CN()


def get_cfg_defaults():
    return _C.clone()
