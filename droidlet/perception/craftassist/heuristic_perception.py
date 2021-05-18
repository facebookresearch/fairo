"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import heapq
import math
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.optimize import linprog
from copy import deepcopy

import logging
from droidlet.shared_data_struct import craftassist_specs as minecraft_specs

from droidlet.lowlevel.minecraft.mc_util import (
    manhat_dist,
    get_locs_from_entity,
    build_safe_diag_adjacent,
    euclid_dist,
    to_block_pos,
    fill_idmeta,
)
from droidlet.lowlevel.minecraft.block_data import BORING_BLOCKS, PASSABLE_BLOCKS, COLOR_BID_MAP
from droidlet.perception.craftassist.search import depth_first_search
from droidlet.memory.craftassist.mc_memory_nodes import InstSegNode, BlockObjectNode

GROUND_BLOCKS = [1, 2, 3, 7, 8, 9, 12, 79, 80]
MAX_RADIUS = 20
BLOCK_DATA = minecraft_specs.get_block_data()
COLOUR_DATA = minecraft_specs.get_colour_data()
BLOCK_PROPERTY_DATA = minecraft_specs.get_block_property_data()
MOB_PROPERTY_DATA = minecraft_specs.get_mob_property_data()
BID_COLOR_DATA = minecraft_specs.get_bid_to_colours()
COLOUR_LIST = list(COLOR_BID_MAP.keys())

# Taken from : stackoverflow.com/questions/16750618/
# whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(points, x):
    """Check if x is in the convex hull of points"""
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def all_nearby_objects(get_blocks, pos, max_radius=MAX_RADIUS):
    """Return a list of connected components near pos.

    Each component is a list of ((x, y, z), (id, meta))

    i.e. this function returns list[list[((x, y, z), (id, meta))]]
    """
    pos = np.round(pos).astype("int32")
    mask, off, blocks = all_close_interesting_blocks(get_blocks, pos, max_radius)
    components = connected_components(mask)
    logging.debug("all_nearby_objects found {} objects near {}".format(len(components), pos))
    xyzbms = [
        [((c[2] + off[2], c[0] + off[0], c[1] + off[1]), tuple(blocks[c])) for c in component_yzxs]
        for component_yzxs in components
    ]
    return xyzbms


def closest_nearby_object(get_blocks, pos):
    """Find the closest interesting object to pos

    Returns a list of ((x,y,z), (id, meta)), or None if no interesting objects are nearby
    """
    objects = all_nearby_objects(get_blocks, pos)
    if len(objects) == 0:
        return None
    centroids = [np.mean([pos for (pos, idm) in obj], axis=0) for obj in objects]
    dists = [manhat_dist(c, pos) for c in centroids]
    return objects[np.argmin(dists)]


def all_close_interesting_blocks(get_blocks, pos, max_radius=MAX_RADIUS):
    """Find all "interesting" blocks close to pos, within a max_radius"""
    mx, my, mz = pos[0] - max_radius, pos[1] - max_radius, pos[2] - max_radius
    Mx, My, Mz = pos[0] + max_radius, pos[1] + max_radius, pos[2] + max_radius

    yzxb = get_blocks(mx, Mx, my, My, mz, Mz)
    relpos = pos - [mx, my, mz]
    mask = accessible_interesting_blocks(yzxb[:, :, :, 0], relpos)
    return mask, (my, mz, mx), yzxb


def accessible_interesting_blocks(blocks, pos):
    """Return a boolean mask of blocks that are accessible-interesting from pos.

    A block b is accessible-interesting if it is
    1. interesting, AND
    2. there exists a path from pos to b through only passable or interesting blocks
    """
    passable = np.isin(blocks, PASSABLE_BLOCKS)
    interesting = np.isin(blocks, BORING_BLOCKS, invert=True)
    passable_or_interesting = passable | interesting
    X = np.zeros_like(passable)

    def _fn(p):
        if passable_or_interesting[p]:
            X[p] = True
            return True
        return False

    depth_first_search(blocks.shape[:3], pos, _fn)
    return X & interesting


def find_closest_component(mask, relpos):
    """Find the connected component of nonzeros that is closest to loc

    Args:
    - mask is a 3d array
    - relpos is a relative position in the mask, with the same ordering

    Returns: a list of indices of the closest connected component, or None
    """
    components = connected_components(mask)
    if len(components) == 0:
        return None
    centroids = [np.mean(cs, axis=0) for cs in components]
    dists = [manhat_dist(c, relpos) for c in centroids]
    return components[np.argmin(dists)]


def connected_components(X, unique_idm=False):
    """Find all connected nonzero components in a array X.
    X is either rank 3 (volume) or rank 4 (volume-idm)
    If unique_idm == True, different block types are different
    components

    Returns a list of lists of indices of connected components
    """
    visited = np.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype="bool")
    components = []
    current_component = set()
    diag_adj = build_safe_diag_adjacent([0, X.shape[0], 0, X.shape[1], 0, X.shape[2]])

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=3)

    def is_air(X, i, j, k):
        return X[i, j, k, 0] == 0

    if not unique_idm:

        def _build_fn(X, current_component, idm):
            def _fn(p):
                if X[p[0], p[1], p[2], 0]:
                    current_component.add(p)
                    return True

            return _fn

    else:

        def _build_fn(X, current_component, idm):
            def _fn(p):
                if tuple(X[p]) == idm:
                    current_component.add(p)
                    return True

            return _fn

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if visited[i, j, k]:
                    continue
                visited[i, j, k] = True
                if is_air(X, i, j, k):
                    continue
                # found a new component
                pos = (i, j, k)
                _fn = _build_fn(X, current_component, tuple(X[i, j, k, :]))
                visited |= depth_first_search(X.shape[:3], pos, _fn, diag_adj)
                components.append(list(current_component))
                current_component.clear()

    return components


def check_between(entities, fat_scale=0.2):
    """Heuristic check if entities[0] is between entities[1] and entities[2]
    by checking if the locs of enitity[0] are in the convex hull of
    union of the max cardinal points of entity[1] and entity[2]"""
    locs = []
    means = []
    for e in entities:
        l = get_locs_from_entity(e)
        if l is not None:
            locs.append(l)
            means.append(np.mean(l, axis=0))
        else:
            # this is not a thing we know how to assign 'between' to
            return False
    mean_separation = euclid_dist(means[1], means[2])
    fat = fat_scale * mean_separation
    bounding_locs = []
    for l in locs:
        if len(l) > 1:
            bl = []
            idx = np.argmax(l, axis=0)
            for i in range(3):
                f = np.zeros(3)
                f[i] = fat
                bl.append(np.array(l[idx[i]]) + fat)
            idx = np.argmin(l, axis=0)
            for i in range(3):
                f = np.zeros(3)
                f[i] = fat
                bl.append(np.array(l[idx[i]]) - fat)
            bounding_locs.append(np.vstack(bl))
        else:
            bounding_locs.append(np.array(l))
    x = np.mean(bounding_locs[0], axis=0)
    points = np.vstack([bounding_locs[1], bounding_locs[2]])
    return in_hull(points, x)


def find_between(entities):
    """Heurisitc search for points between entities[0] and entities[1]
    for now : just pick the point half way between their means
    TODO: fuzz a bit if target is unreachable"""
    for e in entities:
        means = []
        l = get_locs_from_entity(e)
        if l is not None:
            means.append(np.mean(l, axis=0))
        else:
            # this is not a thing we know how to assign 'between' to
            return None
        return (means[0] + means[1]) / 2


def check_inside(entities):
    """Heuristic check on whether an entity[0] is inside entity[1]
    if in some 2d slice, cardinal rays cast from some point in
    entity[0] all hit a block in entity[1], we say entity[0] is inside
    entity[1].  This allows an entity to be inside a ring or
    an open cylinder. This will fail for a "diagonal" ring.
    TODO: "enclosed", where the object is inside in the topological sense"""
    locs = []
    for e in entities:
        l = get_locs_from_entity(e)
        if l is not None:
            locs.append(l)
        else:
            # this is not a thing we know how to assign 'inside' to
            return False
    for b in locs[0]:
        for i in range(3):
            inside = True
            coplanar = [c for c in locs[1] if c[i] == b[i]]
            for j in range(2):
                fixed = (i + 2 * j - 1) % 3
                to_check = (i + 1 - 2 * j) % 3
                colin = [c[to_check] for c in coplanar if c[fixed] == b[fixed]]
                if len(colin) == 0:
                    inside = False
                else:
                    if max(colin) <= b[to_check] or min(colin) >= b[to_check]:
                        inside = False
            if inside:
                return True
    return False


def find_inside(entity):
    """Return a point inside the entity if it can find one.
    TODO: heuristic quick check to find that there aren't any,
    and maybe make this not d^3"""

    # is this a negative object? if yes, just return its mean:
    if hasattr(entity, "blocks"):
        if all(b == (0, 0) for b in entity.blocks.values()):
            m = np.mean(list(entity.blocks.keys()), axis=0)
            return [to_block_pos(m)]
    l = get_locs_from_entity(entity)
    if l is None:
        return []
    m = np.round(np.mean(l, axis=0))
    maxes = np.max(l, axis=0)
    mins = np.min(l, axis=0)
    inside = []
    for x in range(mins[0], maxes[0] + 1):
        for y in range(mins[1], maxes[1] + 1):
            for z in range(mins[2], maxes[2] + 1):
                if check_inside([(x, y, z), entity]):
                    inside.append((x, y, z))
    return sorted(inside, key=lambda x: euclid_dist(x, m))


def label_top_bottom_blocks(block_list, top_heuristic=15, bottom_heuristic=25):
    """This function takes in a list of blocks, where each block is :
    [[x, y, z], id] or [[x, y, z], [id, meta]]
    and outputs a dict:
    {
    "top" : [list of blocks],
    "bottom" : [list of blocks],
    "neither" : [list of blocks]
    }

    The heuristic being used here is : The top blocks are within top_heuristic %
    of the topmost block and the bottom blocks are within bottom_heuristic %
    of the bottommost block.

    Every other block in the list belongs to the category : "neither"
    """
    if type(block_list) is tuple:
        block_list = list(block_list)

    # Sort the list on z, y, x in decreasing order, to order the list
    # to top-down.
    block_list.sort(key=lambda x: (x[0][2], x[0][1], x[0][0]), reverse=True)

    num_blocks = len(block_list)

    cnt_top = math.ceil((top_heuristic / 100) * num_blocks)
    cnt_bottom = math.ceil((bottom_heuristic / 100) * num_blocks)
    cnt_remaining = num_blocks - (cnt_top + cnt_bottom)

    dict_top_bottom = {}
    dict_top_bottom["top"] = block_list[:cnt_top]
    dict_top_bottom["bottom"] = block_list[-cnt_bottom:]
    dict_top_bottom["neither"] = block_list[cnt_top : cnt_top + cnt_remaining]

    return dict_top_bottom


def ground_height(agent, pos, radius, yfilt=5, xzfilt=5):
    """Compute height of ground blocks.
    At the moment: heuristic method and can potentially be replaced with a learned model.
    Can definitely make more sophisticated looks for the first stack of
    non-ground material hfilt high, and can be fooled by e.g. a floating pile of dirt or a big buried object
    """
    ground = np.array(GROUND_BLOCKS).astype("int32")
    offset = yfilt // 2
    yfilt = np.ones(yfilt, dtype="int32")
    L = agent.get_blocks(
        pos[0] - radius, pos[0] + radius, 0, pos[1] + 80, pos[2] - radius, pos[2] + radius
    )
    C = L.copy()
    C = C[:, :, :, 0].transpose([2, 0, 1]).copy()
    G = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(C.shape[0]):
        for j in range(C.shape[2]):
            stack = C[i, :, j].squeeze()
            inground = np.isin(stack, ground) * 1
            inground = np.convolve(inground, yfilt, mode="same")
            G[i, j] = np.argmax(inground == 0)  # fixme what if there isn't one

    G = median_filter(G, size=xzfilt)
    return G - offset


def get_nearby_airtouching_blocks(agent, location, radius=15):
    """Get all blocks in 'radius' of 'location'
    that are touching air on either side.
    Returns:
        A list of blocktypes
    """
    gh = ground_height(agent, location, 0)[0, 0]
    x, y, z = location
    ymin = int(max(y - radius, gh))
    yzxb = agent.get_blocks(x - radius, x + radius, ymin, y + radius, z - radius, z + radius)
    xyzb = yzxb.transpose([2, 0, 1, 3]).copy()
    components = connected_components(xyzb, unique_idm=True)
    blocktypes = []
    for c in components:
        tags = None
        for loc in c:
            idm = tuple(xyzb[loc[0], loc[1], loc[2], :])
            for coord in range(3):
                for d in [-1, 1]:
                    off = [0, 0, 0]
                    off[coord] = d
                    l = (loc[0] + off[0], loc[1] + off[1], loc[2] + off[2])
                    if l[coord] >= 0 and l[coord] < xyzb.shape[coord]:
                        if xyzb[l[0], l[1], l[2], 0] == 0:
                            try:
                                blocktypes.append(idm)
                                type_name = BLOCK_DATA["bid_to_name"][idm]
                                tags = [type_name]
                                colours = deepcopy(
                                    COLOUR_DATA["name_to_colors"].get(type_name, [])
                                )
                                colours.extend([c for c in COLOUR_LIST if c in type_name])
                                if colours:
                                    tags.extend(colours)
                                    tags.extend([{"has_colour": c} for c in colours])
                                tags.extend(
                                    BLOCK_PROPERTY_DATA["name_to_properties"].get(type_name, [])
                                )
                            except:
                                logging.debug(
                                    "I see a weird block, ignoring: ({}, {})".format(
                                        idm[0], idm[1]
                                    )
                                )
        if tags:
            shifted_c = [(l[0] + x - radius, l[1] + ymin, l[2] + z - radius) for l in c]
            if len(shifted_c) > 0:
                InstSegNode.create(agent.memory, shifted_c, tags=tags)
    return blocktypes


def get_all_nearby_holes(agent, location, radius=15, store_inst_seg=True):
    """Returns:
    a list of holes. Each hole is an InstSegNode"""
    sx, sy, sz = location
    max_height = sy + 5
    map_size = radius * 2 + 1
    height_map = [[sz] * map_size for i in range(map_size)]
    hid_map = [[-1] * map_size for i in range(map_size)]
    idm_map = [[(0, 0)] * map_size for i in range(map_size)]
    visited = set([])
    global current_connected_comp, current_idm
    current_connected_comp = []
    current_idm = (2, 0)

    # helper functions
    def get_block_info(x, z):  # fudge factor 5
        height = max_height
        while True:
            B = agent.get_blocks(x, x, height, height, z, z)
            if (
                (B[0, 0, 0, 0] != 0)
                and (x != sx or z != sz or height != sy)
                and (x != agent.pos[0] or z != agent.pos[2] or height != agent.pos[1])
                and (B[0, 0, 0, 0] != 383)
            ):  # if it's not a mobile block (agent, speaker, mobs)
                return height, tuple(B[0, 0, 0])
            height -= 1

    gx = [0, 0, -1, 1]
    gz = [1, -1, 0, 0]

    def dfs(x, y, z):
        """Traverse current connected component and return minimum
        height of all surrounding blocks"""
        build_height = 100000
        if (x, y, z) in visited:
            return build_height
        global current_connected_comp, current_idm
        current_connected_comp.append((x - radius + sx, y, z - radius + sz))  # absolute positions
        visited.add((x, y, z))
        for d in range(4):
            nx = x + gx[d]
            nz = z + gz[d]
            if nx >= 0 and nz >= 0 and nx < map_size and nz < map_size:
                if height_map[x][z] == height_map[nx][nz]:
                    build_height = min(build_height, dfs(nx, y, nz))
                else:
                    build_height = min(build_height, height_map[nx][nz])
                    current_idm = idm_map[nx][nz]
            else:
                # bad ... hole is not within defined radius
                return -100000
        return build_height

    # find all holes
    blocks_queue = []
    for i in range(map_size):
        for j in range(map_size):
            height_map[i][j], idm_map[i][j] = get_block_info(i - radius + sx, j - radius + sz)
            heapq.heappush(blocks_queue, (height_map[i][j] + 1, (i, height_map[i][j] + 1, j)))
    holes = []
    while len(blocks_queue) > 0:
        hxyz = heapq.heappop(blocks_queue)
        h, (x, y, z) = hxyz  # NB: relative positions
        if (x, y, z) in visited or y > max_height:
            continue
        assert h == height_map[x][z] + 1, " h=%d heightmap=%d, x,z=%d,%d" % (
            h,
            height_map[x][z],
            x,
            z,
        )  # sanity check
        current_connected_comp = []
        current_idm = (2, 0)
        build_height = dfs(x, y, z)
        if build_height >= h:
            holes.append((current_connected_comp.copy(), current_idm))
            cur_hid = len(holes) - 1
            for n, xyz in enumerate(current_connected_comp):
                x, y, z = xyz
                rx, ry, rz = x - sx + radius, y + 1, z - sz + radius
                heapq.heappush(blocks_queue, (ry, (rx, ry, rz)))
                height_map[rx][rz] += 1
                if hid_map[rx][rz] != -1:
                    holes[cur_hid][0].extend(holes[hid_map[rx][rz]][0])
                    holes[hid_map[rx][rz]] = ([], (0, 0))
                hid_map[rx][rz] = cur_hid

    # A bug in the algorithm above produces holes that include non-air blocks.
    # Just patch the problem here, since this function will eventually be
    # performed by an ML model
    for i, (xyzs, idm) in enumerate(holes):
        blocks = fill_idmeta(agent, xyzs)
        xyzs = [xyz for xyz, (d, _) in blocks if d == 0]  # remove non-air blocks
        holes[i] = (xyzs, idm)

    # remove 0-length holes
    holes = [h for h in holes if len(h[0]) > 0]
    hole_mems = []
    for hole in holes:
        memid = InstSegNode.create(agent.memory, hole[0], tags=["hole", "pit", "mine"])
        try:
            fill_block_name = BLOCK_DATA["bid_to_name"][hole[1]]
        except:
            idm = (hole[1][0], 0)
            fill_block_name = BLOCK_DATA["bid_to_name"].get(idm)
        if fill_block_name:
            fill_block_mems = agent.memory.basic_search(
                {
                    "base_table": "BlockTypes",
                    "triples": [{"pred_text": "has_name", "obj_text": fill_block_name}],
                }
            )
            fill_block_memid = fill_block_mems[0].memid
            agent.memory.add_triple(subj=memid, pred_text="has_fill_type", obj=fill_block_memid)
        hole_mems.append(agent.memory.get_mem_by_id(memid))
    return hole_mems


def maybe_get_type_name(idm):
    try:
        type_name = BLOCK_DATA["bid_to_name"][idm]
    except:
        type_name = "UNK"
        logging.debug(
            "heuristic perception encountered unknown block: ({}, {})".format(idm[0], idm[1])
        )
    return type_name


class PerceptionWrapper:
    """Perceive the world at a given frequency and update agent
    memory.

    | Runs three basic heuristics.
    | The first finds "interesting" visible (e.g. touching air) blocks, and
    |     creates InstSegNodes
    | The second finds "holes", and creates InstSegNodes
    | The third finds connected components of "interesting" blocks and creates
    |     BlockObjectNodes.  Note that this only places something in memory if
    |     the connected component is newly closer than 15 blocks of the agent,
    |     but was not recently placed (if it were newly visible because of
    |     a block placement, it would be dealt with via maybe_add_block_to_memory
    |     in low_level_perception.py)

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        perceive_freq (int): if not forced, how many Agent steps between perception
    """

    def __init__(self, agent, perceive_freq=20):
        self.perceive_freq = perceive_freq
        self.agent = agent
        self.radius = 15

    def perceive(self, force=False):
        """Called by the core event loop for the agent to run all perceptual
        models and save their state to memory.

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        if self.perceive_freq == 0 and not force:
            return
        if self.agent.count % self.perceive_freq != 0 and not force:
            return
        if force or not self.agent.memory.task_stack_peek():
            # perceive blocks in marked areas
            for pos, radius in self.agent.areas_to_perceive:
                for obj in all_nearby_objects(self.agent.get_blocks, pos, radius):
                    memid = BlockObjectNode.create(self.agent.memory, obj)
                    color_tags = []
                    for idm in obj:
                        type_name = maybe_get_type_name(idm)
                        color_tags.extend(COLOUR_DATA["name_to_colors"].get(type_name, []))
                    for color_tag in list(set(color_tags)):
                        self.agent.memory.add_triple(
                            subj=memid, pred_text="has_colour", obj_text=color_tag
                        )

                get_all_nearby_holes(self.agent, pos, radius)
                get_nearby_airtouching_blocks(self.agent, pos, radius)

            # perceive blocks near the agent
            for objs in all_nearby_objects(self.agent.get_blocks, self.agent.pos):
                memid = BlockObjectNode.create(self.agent.memory, objs)
                color_tags = []
                for obj in objs:
                    idm = obj[1]
                    type_name = maybe_get_type_name(idm)
                    color_tags.extend(COLOUR_DATA["name_to_colors"].get(type_name, []))
                for color_tag in list(set(color_tags)):
                    self.agent.memory.add_triple(
                        subj=memid, pred_text="has_colour", obj_text=color_tag
                    )

            get_all_nearby_holes(self.agent, self.agent.pos, radius=self.radius)
            get_nearby_airtouching_blocks(self.agent, self.agent.pos, radius=self.radius)
