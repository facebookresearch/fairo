"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np


def no_y_l1(self, xyz, k):
    """ returns the l1 distance between two standard coordinates"""
    return np.linalg.norm(np.asarray([xyz[0], xyz[2]]) - np.asarray([k[0], k[2]]), ord=1)


class PlaceField:
    """
    maintains a grid-based map of a slice of the world, and 
    the state representations needed to track active exploration.

    droidlet.interpreter.robot.tasks.CuriousExplore uses the can_examine method to decide 
    which objects to explore next:
    1. for each new candidate coordinate, it fetches the closest examined coordinate.
    2. if this closest coordinate is within a certain threshold (1 meter) of the current coordinate, 
    or if that region has been explored upto a certain number of times (2, for redundancy),
    it is not explored, since a 'close-enough' region in space has already been explored. 
    """

    examined = {}
    examined_id = set()
    last = None

    def get_closest(self, xyz):
        """returns closest examined point to xyz"""
        c = None
        dist = 1.5
        for k, v in self.examined.items():
            if no_y_l1(k, xyz) < dist:
                dist = no_y_l1(k, xyz)
                c = k
        if c is None:
            self.examined[xyz] = 0
            return xyz
        return c

    def update(self, target):
        """called each time a region is examined. Updates relevant states."""
        self.last = self.get_closest(target["xyz"])
        self.examined_id.add(target["eid"])
        self.examined[self.last] += 1

    def clear_examined(self):
        self.examined = {}
        self.examined_id = set()
        self.last = None

    def can_examine(self, x):
        """decides whether to examine x or not."""
        loc = x["xyz"]
        k = self.get_closest(x["xyz"])
        val = True
        if self.last is not None and self.l1(cls.last, k) < 1:
            val = False
        val = self.examined[k] < 2
        print(
            f"can_examine {x['eid'], x['label'], x['xyz'][:2]}, closest {k[:2]}, can_examine {val}"
        )
        print(f"examined[k] = {self.examined[k]}")
        return val
