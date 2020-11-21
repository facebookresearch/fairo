"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from collections import namedtuple
from unittest.mock import Mock


Player = namedtuple("Player", "entityId, name, pos, look, mainHand")
Mob = namedtuple("Mob", "entityId, mobType, pos, look")
Pos = namedtuple("Pos", "x, y, z")
Look = namedtuple("Look", "yaw, pitch")
Item = namedtuple("Item", "id, meta")


class PickleMock(Mock):
    """Mocks cannot be pickled. This Mock class can."""

    def __reduce__(self):
        return (Mock, ())
