"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np

MAX_MAP_SIZE = 8193
MAP_INIT_SIZE = 1025
BIG_I = MAX_MAP_SIZE
BIG_J = MAX_MAP_SIZE


def no_y_l1(self, xyz, k):
    """returns the l1 distance between two standard coordinates"""
    return np.linalg.norm(np.asarray([xyz[0], xyz[2]]) - np.asarray([k[0], k[2]]), ord=1)


# TODO tighter integration with reference objects table, main memory update
# should probably sync PlaceField maps without explicit perception updates
# Node type for complicated-shaped obstacles that aren't "objects" e.g. walls?
#    currently just represented as occupancy cells with no memid
# FIXME allow multiple memids at a single location in the map


class PlaceField:
    """
    maintains a grid-based map of some slice(s) of the world, and
    the state representations needed to track active exploration.

    the .place_fields attribute is a dict with keys corresponding to heights,
    and values {"map": 2d numpy array, "updated": 2d numpy array, "memids": 2d numpy array}
    place_fields[h]["map"] is an occupany map at the the height h (in agent coordinates)
                           a location is 0 if it is traversible or it is unseen, 1 if occupied
                           by a non-traversible obstacle
    place_fields[h]["memids"] gives a memid index for the ReferenceObject at that location,
                              if there is a ReferenceObject linked to that spatial location.
                              the PlaceField keeps a mappping from the indices to memids in
                              self.index2memid and self.memid2index
    place_fields[h]["updated"] gives the last update time of that location (in agent's internal time)
                               if -1, it has neer been updated

    the .map2real method converts a location from a map to world coords
    the .real2map method converts a location from the world to the map coords

    the traversibility map is not currently coupled to the occupancy map

    droidlet.interpreter.robot.tasks.CuriousExplore uses the can_examine method to decide
    which objects to explore next:
    1. for each new candidate coordinate, it fetches the closest examined coordinate.
    2. if this closest coordinate is within a certain threshold (1 meter) of the current coordinate,
    or if that region has been explored upto a certain number of times (2, for redundancy),
    it is not explored, since a 'close-enough' region in space has already been explored.
    """

    def __init__(self, memory, pixels_per_unit=1):
        self.get_time = memory.get_time

        self.index2memid = []
        self.memid2index = {}

        self.examined = {}
        self.examined_id = set()
        self.last = None

        self.maps = {}
        self.maybe_add_memid("NULL")
        self.maybe_add_memid(memory.self_memid)
        # FIXME, want slices, esp for mc... init after first perception
        # with h=y2slice(y) instead of using 0
        self.map_size = self.extend_map(h=0)

        self.pixels_per_unit = pixels_per_unit

        # gives an index allowing quick lookup by memid
        # each entry is keyed by a memid and is a dict
        # {str(h*BIG_I*BIG_J + i*BIG_J + j) : Traversible}
        # for each placed h, i ,j
        self.memid2locs = {}

    def ijh2idx(self, i, j, h):
        return str(h * BIG_I * BIG_J + i * BIG_J + j)

    def idx2ijh(self, idx):
        idx = int(idx)
        j = idx % BIG_J
        idx = (idx - j) // BIG_J
        i = idx % BIG_I
        h = (idx - i) // BIG_I
        return i, j, h

    def pop_memid_loc(self, memid, i, j, h):
        idx = self.hij2idx(h, i, j)
        del self.memid2locs[memid][idx]

    def maybe_delete_loc(self, i, j, h, t, memid="NULL"):
        """
        remove a loc from the maps and from memid2loc index.
        if memid is set, only removes the loc if the memid matches
        """
        current_memid = self.index2memid[int(self.maps[h]["memids"][i, j])]
        if memid == "NULL" or current_memid == memid:
            self.maps[h]["memids"][i, j] = self.memid2index["NULL"]
            self.maps[h]["map"][i, j] = 0
            self.maps[h]["updated"][i, j] = t
            idx = self.ijh2idx(i, j, h)
            # maybe error/warn if its not there?
            if self.memid2locs.get(memid):
                self.memid2locs[memid].pop(idx, None)
                if len(self.memid2locs[memid]) == 0:
                    self.memid2locs.pop(memid, None)

    # this is pretty brutal, using rn on robot to sync with slam service.
    # overrides slam service for "detections" with memids that have been labeled
    # as obstacles (e.g. self_memid)
    def sync_traversible(self, locs, h=0):
        # overwrite traversibility map from slam service
        self.maps[h]["map"][:] = 0
        self.maps[h]["updated"][:] = self.get_time()
        for x, z in locs:
            i, j = self.real2map(x, z, h)
            s = max(i - self.map_size + 1, j - self.map_size + 1, -i, -j)
            if s > 0:
                self.extend_map(h=h, extension=s)
                i, j = self.real2map(x, z, h)
            self.maps[h]["map"][i, j] = 1
        # replace memids that are obstacles if they were clobbered from map
        for k, v in self.memid2locs.items():
            for idx, t in v.items():
                i, j, height = self.idx2ijh(idx)
                if height == h and t > 0:
                    self.maps[h]["map"][i, j] = 1

    def get_obstacle_list(self):
        """
        outputs a list of all impassable locations in agent coordinates.
        """
        # FIXME! do other h
        ijs = list(zip(*self.maps[0]["map"].nonzero()))
        return [self.map2real(i, j, 0) for i, j in ijs]

    def delete_loc_by_memid(self, memid, t, is_move=False):
        """
        remove all locs corresponding to a memid.
        if is_move is set, asserts that there is precisely one loc
        corresponding to the memid
        """
        assert memid
        assert memid != "NULL"
        count = 0
        for idx in self.memid2locs.get(memid, []):
            i, j, h = self.idx2ijh(idx)
            self.maps[h]["memids"][i, j] = 0
            # FIXME: maybe this loc is still not traversible...
            self.maps[h]["map"][i, j] = 0
            self.maps[h]["updated"][i, j] = t
            count = count + 1
            if is_move and count > 1:
                # eventually allow moving "large" objects
                raise Exception(
                    "tried to delete more than one pixel from the place_field by memid with is_move set"
                )
        self.memid2locs.pop(memid, None)

    def update_map(self, changes):
        """
        changes is a list of dicts of the form
        {"pos": (x, y, z),
        "memid": str (default "NULL"),
        "is_obstacle": bool (default True),
        "is_move": bool (default False),
        "is_delete": bool (default False) }
        pos is required if is_delete is False.
        all other fields are always optional.

        "is_obstacle" tells whether the agent can traverse that location
        if "is_move" is False, the change is taken as is; if "is_move" is True, if the
            change corresponds to a memid, and the memid is located somewhere on the map,
            the old location is removed when the new one is set.  For now, to move complicated objects
            that cover many pixels, do not use is_move, and instead move them "by hand"
            by issuing a list of changes deleting the old now empty locations and adding the
            new now-filled locations
        "is_delete" True without a memid means whatever is in that location is to be removed.
            if a memid is set, the remove will occur only if the memid matches.

        the "is_obstacle" status can be changed without changing memid etc.
        """
        t = self.get_time()
        for c in changes:
            is_delete = c.get("is_delete", False)
            is_move = c.get("is_move", False)
            memid = c.get("memid", "NULL")
            p = c.get("pos")
            if p is None:
                assert is_delete
                # if the change is a remove, and is specified by memid:
                if not memid:
                    raise Exception("tried to update a map location without a location or a memid")
                # warn if empty TODO?
                self.delete_loc_by_memid(memid, t)
            else:
                x, y, z = p
                h = self.y2slice(y)
                i, j = self.real2map(x, z, h)
                s = max(i - self.map_size + 1, j - self.map_size + 1, -i, -j)
                if s > 0:
                    self.extend_map(extension=s)
                i, j = self.real2map(x, z, h)
                s = max(i - self.map_size + 1, j - self.map_size + 1, -i, -j)
                if s > 0:
                    # the map can not been extended enough to handle these bc MAX_MAP_SIZE
                    # FIXME appropriate warning or error?
                    continue
                if is_delete:
                    self.maybe_delete_loc(i, j, h, t, memid=memid)
                else:
                    if is_move:
                        assert memid != "NULL"
                        self.delete_loc_by_memid(memid, t, is_move=True)
                    self.maps[h]["memids"][i, j] = self.memid2index.get(
                        memid, self.maybe_add_memid(memid)
                    )
                    self.maps[h]["map"][i, j] = c.get("is_obstacle", 1)
                    self.maps[h]["updated"][i, j] = t
                    if not self.memid2locs.get(memid):
                        self.memid2locs[memid] = {}
                    self.memid2locs[memid][self.ijh2idx(i, j, h)] = c.get("is_obstacle", 1)

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
        i = i + n // 2
        j = j + n // 2
        return round(i), round(j)

    def map2real(self, i, j, h):
        """
        convert an i, j pixel coordinate in the map to agent space
        """
        n = self.maps[h]["map"].shape[0]
        i = i - n // 2
        j = j - n // 2
        x = i / self.pixels_per_unit
        z = j / self.pixels_per_unit
        return x, z

    def maybe_add_memid(self, memid):
        """
        adds an entry to the mapping from memids to ints to put on map.
        these are never removed
        """
        idx = self.memid2index.get(memid)
        if idx is None:
            idx = len(self.index2memid)
            self.index2memid.append(memid)
            self.memid2index[memid] = idx
        return idx

    def extend_map(self, h=None, extension=1):
        assert extension >= 0
        if not h and len(self.maps) == 1:
            h = list(self.maps.keys())[0]
        if not self.maps.get(h):
            self.maps[h] = {}
            for m, v in {"updated": -1, "map": 0, "memids": 0}.items():
                self.maps[h][m] = v * np.ones((MAP_INIT_SIZE, MAP_INIT_SIZE))
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


class EmptyPlaceField:
    """
    this will be used as a dummy instead of regular PlaceField
    """

    def sync_traversible(self, locs, h=0):
        pass

    def update_map(self, changes):
        pass

    def get_closest(self):
        return None

    def get_obstacle_list(self):
        return []


if __name__ == "__main__":
    W = {0: {0: {0: True}, 1: {2: {3: True}}}, 1: {5: True}}
    idxs = [0, 1, 2, 3]
