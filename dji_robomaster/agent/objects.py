"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from collections import namedtuple


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


Pos = namedtuple("pos", ["x", "y", "z"])
Properties = namedtuple("properties", ["x", "y", "z"])
Marker = namedtuple("Marker", "markerId pos color category properties")
