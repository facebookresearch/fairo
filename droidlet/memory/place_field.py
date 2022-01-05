"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np

MAX_MAP_SIZE = 4097
MAP_INIT_SIZE = 1025


def no_y_l1(self, xyz, k):
    """ returns the l1 distance between two standard coordinates"""
    return np.linalg.norm(np.asarray([xyz[0], xyz[2]]) - np.asarray([k[0], k[2]]), ord=1)


class PlaceField:
    """
    maintains a grid-based map of some slice(s) of the world, and 
    the state representations needed to track active exploration.

    the .place_fields attribute is a dict with keys corresponding to heights, 
    and values {"map": 2d numpy array, "updated": 2d numpy array, "memids": 2d numpy array}
    place_fields[h]["map"] is an occupany map at the the height h (in agent coordinates)
                           a location is 0 if there is nothing there or it is unseen, 1 if occupied  
    place_fields[h]["memids"] gives a memid index for the ReferenceObject at that location, 
                              if there is a ReferenceObject linked to that spatial location.
                              the PlaceField keeps a mappping from the indices to memids in 
                              self.index2memid and self.memid2index
    place_fields[h]["updated"] gives the last update time of that location (in agent's internal time)
                               if -1, it has neer been updated

    the .map2real method converts a location from a map to world coords
    the .real2map method converts a location from the world to the map coords

    droidlet.interpreter.robot.tasks.CuriousExplore uses the can_examine method to decide 
    which objects to explore next:
    1. for each new candidate coordinate, it fetches the closest examined coordinate.
    2. if this closest coordinate is within a certain threshold (1 meter) of the current coordinate, 
    or if that region has been explored upto a certain number of times (2, for redundancy),
    it is not explored, since a 'close-enough' region in space has already been explored. 
    """

    # FIXME allow multiple memids at a single location in the map
    def __init__(self, memory, pixels_per_unit=1):
        self.get_time = memory.get_time

        self.index2memid = []
        self.memid2index = {}

        self.examined = {}
        self.examined_id = set()
        self.last = None

        self.maps = {}
        self.add_memid("NULL")
        self.add_memid(memory.self_memid)
        self.map_size = self.extend_map()

        self.pixels_per_unit = pixels_per_unit

    def update_map(self, changes):
        """
        changes is a list of tuples of the form (x, y, z, memid, is_obstacle)
        giving the obstacles in that slice, the memid of the object at that location
        if there is a memid ("NULL" otherwise), and is_obstacle if its blocking.  An object need not be 
        blocking to have a memid.  an obstacle can be cleared by inputting the proper tuple.
        in world coordinates. is_obstacle is 1 for obstacle and 0 for free space
        """
        t = self.get_time()
        for x, y, z, memid, is_obstacle in changes:
            h = self.y2slice(y)
            i, j = self.real2map(x, z)
            s = max(i - self.map_size + 1, j - self.map_size + 1, -i, -j)
            if s > 0:
                self.extend_map(s)
            i, j = self.real2map(x, z, h)
            s = max(i - self.map_size + 1, j - self.map_size + 1, -i, -j)
            if s > 0:
                # the map can not been extended enough to handle these bc MAX_MAP_SIZE
                # FIXME appropriate warning or error?
                continue
            self.maps[h]["map"][i, j] = is_obstacle
            idx = self.memid2index.get(memid, self.add_memid(memid))
            self.maps[h]["memids"] = idx
            self.maps[k]["updated"] = t

    # FIXME, want slices, esp for mc
    def y2slice(self, y):
        return 0

    def real2map(self, x, z, h):
        """
        convert an x, z coordinate in agent space to a pixel on the map
        """
        n = self.maps[h]["map"].shape[0]
        i = x * self.pixels_per_unit
        j = z * self.pixels_per_unit
        i = i - n // 2
        j = j - n // 2
        return i, j

    def map2real(self, i, j, h):
        """
        convert an x, z coordinate in agent space to a pixel on the map
        """
        n = self.maps[h]["map"].shape[0]
        i = i + n // 2
        j = j + n // 2
        x = i / self.pixels_per_unit
        z = j / self.pixels_per_unit
        return x, z

    def add_memid(self, memid):
        self.index2memid.append(memid)
        idx = len(self.index2memid)
        self.memid2index[memid] = idx
        return idx

    def extend_map(self, h=None, extension=1):
        assert extension >= 0
        if not h and len(self.maps) == 1:
            h = list(self.maps.keys())[0]
        if not self.maps.get(h):
            self.maps[h] = {
                "map": np.zeros(MAP_INIT_SIZE, MAP_INIT_SIZE),
                "updated": -np.ones((MAP_INIT_SIZE, MAP_INIT_SIZE)),
            }
        w = self.maps[h]["map"].shape[0]
        new_w = w + 2 * extension
        if new_w > MAX_MAP_SIZE:
            return -1
        for m, v in {"updated": -1, "map": 0, "memids": 0}.items():
            new_map = v * np.ones((new_w, new_w))
            new_map[extension:-extension, extension:-extension] = self.maps[h][m]
            self.maps[h][m] = new_map
        return new_w

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
