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
from droidlet.base_util import depth_first_search, to_block_pos, manhat_dist, euclid_dist
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData

GROUND_BLOCKS = [1, 2, 3, 7, 8, 9, 12, 79, 80]
MAX_RADIUS = 20


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


def all_nearby_objects(get_blocks, pos, boring_blocks, passable_blocks, max_radius=MAX_RADIUS):
    """Return a list of connected components near pos.

    Each component is a list of ((x, y, z), (id, meta))

    i.e. this function returns list[list[((x, y, z), (id, meta))]]
    """
    pos = np.round(pos).astype("int32")
    mask, off, blocks = all_close_interesting_blocks(
        get_blocks, pos, boring_blocks, passable_blocks, max_radius
    )
    components = connected_components(mask)
    logging.debug("all_nearby_objects found {} objects near {}".format(len(components), pos))
    xyzbms = [
        [((c[2] + off[2], c[0] + off[0], c[1] + off[1]), tuple(blocks[c])) for c in component_yzxs]
        for component_yzxs in components
    ]
    return xyzbms


def closest_nearby_object(get_blocks, pos, boring_blocks, passable_blocks):
    """Find the closest interesting object to pos

    Returns a list of ((x,y,z), (id, meta)), or None if no interesting objects are nearby
    """
    objects = all_nearby_objects(get_blocks, pos, boring_blocks, passable_blocks)
    if len(objects) == 0:
        return None
    centroids = [np.mean([pos for (pos, idm) in obj], axis=0) for obj in objects]
    dists = [manhat_dist(c, pos) for c in centroids]
    return objects[np.argmin(dists)]


def all_close_interesting_blocks(
    get_blocks, pos, boring_blocks, passable_blocks, max_radius=MAX_RADIUS
):
    """Find all "interesting" blocks close to pos, within a max_radius"""
    mx, my, mz = pos[0] - max_radius, pos[1] - max_radius, pos[2] - max_radius
    Mx, My, Mz = pos[0] + max_radius, pos[1] + max_radius, pos[2] + max_radius

    yzxb = get_blocks(mx, Mx, my, My, mz, Mz)
    relpos = pos - [mx, my, mz]
    mask = accessible_interesting_blocks(yzxb[:, :, :, 0], relpos, boring_blocks, passable_blocks)
    return mask, (my, mz, mx), yzxb


def accessible_interesting_blocks(blocks, pos, boring_blocks, passable_blocks):
    """Return a boolean mask of blocks that are accessible-interesting from pos.

    A block b is accessible-interesting if it is
    1. interesting, AND
    2. there exists a path from pos to b through only passable or interesting blocks
    """
    passable = np.isin(blocks, passable_blocks)
    interesting = np.isin(blocks, boring_blocks, invert=True)
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


def check_between(entities, get_locs_from_entity, fat_scale=0.2):
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


def check_inside(entities, get_locs_from_entity):
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


def find_inside(entity, get_locs_from_entity):
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
                if check_inside([(x, y, z), entity], get_locs_from_entity):
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


def get_nearby_airtouching_blocks(
    agent, location, block_data, color_data, block_property_data, color_list, radius=15
):
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
    all_components = []
    all_tags = []
    for c in components:
        tags = []
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
                                type_name = block_data["bid_to_name"][idm]
                                tags.append(type_name)
                                colours = deepcopy(color_data["name_to_colors"].get(type_name, []))
                                colours.extend([c for c in color_list if c in type_name])
                                if colours:
                                    tags.extend(colours)
                                    tags.extend([{"has_colour": c} for c in colours])
                                tags.extend(
                                    block_property_data["name_to_properties"].get(type_name, [])
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
                all_components.append(shifted_c)
                all_tags.append(tags)
    return blocktypes, all_components, all_tags


def get_all_nearby_holes(agent, location, block_data, fill_idmeta, radius=15, store_inst_seg=True):
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

    # utility functions
    def get_block_info(x, z):  # fudge factor 5
        height = max_height
        min_height = -50

        if agent.backend == "pyworld":
            min_height = -1

        while True and height > min_height:
            B = agent.get_blocks(x, x, height, height, z, z)
            if (
                (B[0, 0, 0, 0] != 0)
                and (x != sx or z != sz or height != sy)
                and (x != agent.pos[0] or z != agent.pos[2] or height != agent.pos[1])
                and (B[0, 0, 0, 0] != 383)
            ):  # if it's not a mobile block (agent, speaker, mobs)
                return height, tuple(B[0, 0, 0])
            height -= 1

        return min_height, (0, 0)

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
    return holes


def maybe_get_type_name(idm, block_data):
    try:
        type_name = block_data["bid_to_name"][idm]
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

    def __init__(self, agent, low_level_data, perceive_freq=20, mark_airtouching_blocks=False):
        self.mark_airtouching_blocks = mark_airtouching_blocks
        self.perceive_freq = perceive_freq
        self.agent = agent
        self.radius = 15
        self.block_data = low_level_data["block_data"]
        self.color_data = low_level_data["color_data"]
        self.block_property_data = low_level_data["block_property_data"]
        self.boring_blocks = low_level_data["boring_blocks"]
        self.passable_blocks = low_level_data["passable_blocks"]
        self.color_bid_map = low_level_data["color_bid_map"]

    def perceive(self, force=False):
        """Called by the core event loop for the agent to run all perceptual
        models and return their state to agent.

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        color_list = list(self.color_bid_map.keys())
        if self.perceive_freq == 0 and not force:
            return CraftAssistPerceptionData()
        if self.agent.count % self.perceive_freq != 0 and not force:
            return CraftAssistPerceptionData()

        """
        perceive_info is a dictionary with the following possible members :
         - in_perceive_area(dict) : member that has following possible children in "areas_to_perceive" -
            - block_object_attributes(list) - List of [obj, color_tags] of all nearby objects
            - holes(list) - List of non-zero length holes where each item in list is (connected_component, idmeta)
            - airtouching_blocks(list) - List of [shifted_coordinates, list of tags]
         - near agent (dict) : member that has following possible children near the agent's location  -
            - block_object_attributes(list) - List of [obj, color_tags] of all nearby objects
            - holes(list) - List of non-zero length holes where each item in list is (connected_component, idmeta)
            - airtouching_blocks(list) - List of [shifted_coordinates, list of tags]
        """
        perceive_info = {}
        perceive_info[
            "in_perceive_area"
        ] = {}  # dictionary with children: block objects and holes in perception area
        perceive_info[
            "near_agent"
        ] = {}  # dictionary with children: block objects and holes near the agent
        # 1. perceive blocks in marked areas to perceive
        for pos, radius in self.agent.areas_to_perceive:
            # 1.1 Get block objects and their colors
            obj_tag_list = []
            for obj in all_nearby_objects(
                self.agent.get_blocks, pos, self.boring_blocks, self.passable_blocks, radius
            ):
                color_tags = []
                for idm in obj:
                    type_name = maybe_get_type_name(idm, self.block_data)
                    color_tags.extend(self.color_data["name_to_colors"].get(type_name, []))
                obj_tag_list.append([obj, color_tags])
            perceive_info["in_perceive_area"]["block_object_attributes"] = (
                obj_tag_list if obj_tag_list else None
            )
            # 1.2 Get all holes in perception area
            holes = get_all_nearby_holes(
                self.agent, pos, self.block_data, self.agent.low_level_data["fill_idmeta"], radius
            )
            perceive_info["in_perceive_area"]["holes"] = holes if holes else None
            # 1.3 Get all air-touching blocks in perception area
            if self.mark_airtouching_blocks:
                blocktypes, shifted_c, tags = get_nearby_airtouching_blocks(
                    self.agent,
                    pos,
                    self.block_data,
                    self.color_data,
                    self.block_property_data,
                    color_list,
                    radius,
                )
                if tags and len(shifted_c) > 0:
                    perceive_info["in_perceive_area"]["airtouching_blocks"] = list(
                        zip(shifted_c, tags)
                    )

        # 2. perceive blocks and their colors near the agent
        near_obj_tag_list = []
        for objs in all_nearby_objects(
            self.agent.get_blocks, self.agent.pos, self.boring_blocks, self.passable_blocks
        ):
            color_tags = []
            for obj in objs:
                idm = obj[1]
                type_name = maybe_get_type_name(idm, self.block_data)
                color_tags.extend(self.color_data["name_to_colors"].get(type_name, []))
            near_obj_tag_list.append([objs, color_tags])
        perceive_info["near_agent"] = perceive_info.get("near_agent", {})
        perceive_info["near_agent"]["block_object_attributes"] = (
            near_obj_tag_list if near_obj_tag_list else None
        )
        # 3. Get all holes near agent
        holes = get_all_nearby_holes(
            self.agent,
            self.agent.pos,
            self.block_data,
            self.agent.low_level_data["fill_idmeta"],
            radius=self.radius,
        )
        perceive_info["near_agent"]["holes"] = holes if holes else None
        # 4. Get all air-touching blocks near agent
        if self.mark_airtouching_blocks:
            blocktypes, shifted_c, tags = get_nearby_airtouching_blocks(
                self.agent,
                self.agent.pos,
                self.block_data,
                self.color_data,
                self.block_property_data,
                color_list,
                radius=self.radius,
            )
            if tags and len(shifted_c) > 0:
                perceive_info["near_agent"]["airtouching_blocks"] = list(zip(shifted_c, tags))

        return CraftAssistPerceptionData(
            in_perceive_area=perceive_info["in_perceive_area"],
            near_agent=perceive_info["near_agent"],
        )


def build_safe_diag_adjacent(bounds):
    """bounds is [mx, Mx, my, My, mz, Mz],
    if nothing satisfies, returns empty list"""

    def a(p):
        """Return the adjacent positions to p including diagonal adjaceny, within the bounds"""
        mx = max(bounds[0], p[0] - 1)
        my = max(bounds[2], p[1] - 1)
        mz = max(bounds[4], p[2] - 1)
        Mx = min(bounds[1] - 1, p[0] + 1)
        My = min(bounds[3] - 1, p[1] + 1)
        Mz = min(bounds[5] - 1, p[2] + 1)
        return [
            (x, y, z)
            for x in range(mx, Mx + 1)
            for y in range(my, My + 1)
            for z in range(mz, Mz + 1)
            if (x, y, z) != p
        ]

    return a
