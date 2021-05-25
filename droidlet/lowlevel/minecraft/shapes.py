"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file has implementation for a variety of shapes and
their arrangements"""
import math
import numpy as np
from typing import Optional, Tuple

IDM = Tuple[int, int]
DEFAULT_IDM = (5, 0)

# TODO cylinder
# TODO: add negative versions of each of the shapes
# TODO shape/schematic object with properties
#     (resizable, symmetries, color patterns allowed, etc.)
# TODO sheet
# TODO rotate
# TODO (in perception) recognize size


def hollow_triangle(
    size=3, depth=1, bid=DEFAULT_IDM, orient="xy", thickness=1, labelme=False, **kwargs
):
    """
    Construct an empty isosceles triangle with a given half base length (b/2).
    The construction is restricted to having stepped blocks for the
    sides other than base. Hence the height is (total base length)//2 + 1.
    """
    assert size > 0
    side = size * 2 + 1  # total base length

    S = []
    L = {}
    insts = {}

    for height in range(side // 2 + 1):
        if height < thickness:
            r = range(height, side - height)
        else:
            r = list(range(height, height + thickness))
            r = r + list(range(side - height - thickness, side - height))
        for i in r:
            if i >= 0:
                for t in range(0, depth):
                    if orient == "xy":
                        S.append(((i, height, t), bid))
                    elif orient == "yz":
                        S.append(((t, i, height), bid))
                    elif orient == "xz":
                        S.append(((i, t, height), bid))

    if not labelme:
        return S
    else:
        return S, L, insts


def hollow_rectangle(
    size: Tuple[int, int] = (5, 3),
    height: Optional[int] = None,
    length: Optional[int] = None,
    bid: IDM = DEFAULT_IDM,
    thickness=1,
    orient="xy",
    labelme=False,
    **kwargs,
):
    R = rectangle(size=size, height=height, length=length, bid=bid, orient=orient)
    l = [r[0] for r in R]
    m = np.min(l, axis=0)
    M = np.max(l, axis=0)

    def close_to_border(r):
        for i in range(3):
            if abs(r[0][i] - m[i]) < thickness or abs(r[0][i] - M[i]) < thickness:
                if M[i] > m[i]:
                    return True
        return False

    S = [r for r in R if close_to_border(r)]
    if not labelme:
        return S
    else:
        return S, {}, {}


def rectangle(
    size=(5, 3),
    height: Optional[int] = None,
    length: Optional[int] = None,
    bid: IDM = DEFAULT_IDM,
    orient="xy",
    labelme=False,
    **kwargs,
):
    if type(size) is int:
        size = [size, size]
    if height is not None:
        size = (height, size[1])
    if length is not None:
        size = (size[0], length)
    assert size[0] > 0
    assert size[1] > 0

    size_n = [size[1], size[0], 1]
    if orient == "yz":
        size_n = [1, size[0], size[1]]
    if orient == "xz":
        size_n = [size[0], 1, size[1]]
    return rectanguloid(size=size_n, bid=bid, labelme=labelme)


def square(size=3, bid=DEFAULT_IDM, orient="xy", labelme=False, **kwargs):
    """Construct square as a rectangle of height and width size"""
    size = [size, size]
    return rectangle(size=size, bid=bid, orient=orient, labelme=labelme, **kwargs)


def triangle(size=3, bid=DEFAULT_IDM, orient="xy", thickness=1, labelme=False, **kwargs):
    """
    Construct an isosceles traingle with a given half base length (b/2).
    The construction is restricted to having stepped blocks for the
    sides other than base. Hence the height is (total base length)//2 + 1.
    """
    assert size > 0
    side = size * 2 + 1  # total base length

    S = []
    L = {}
    insts = {}

    for height in range(side // 2 + 1):
        for i in range(height, side - height):
            for t in range(0, thickness):
                if orient == "xy":
                    S.append(((i, height, t), bid))
                elif orient == "yz":
                    S.append(((t, i, height), bid))
                elif orient == "xz":
                    S.append(((i, t, height), bid))

    if not labelme:
        return S
    else:
        return S, L, insts


def circle(
    radius=4, size=None, bid=DEFAULT_IDM, orient="xy", thickness=1, labelme=False, **kwargs
):
    if size is not None:
        radius = size // 2
    N = 2 * radius
    c = N / 2 - 1 / 2
    S = []
    L = {}
    insts = {}
    tlist = range(0, thickness)
    for r in range(N):
        for s in range(N):
            in_radius = False
            out_core = False
            if ((r - c) ** 2 + (s - c) ** 2) ** 0.5 < N / 2:
                in_radius = True
            if ((r - c) ** 2 + (s - c) ** 2) ** 0.5 > N / 2 - thickness:
                out_core = True
            if in_radius and out_core:
                for t in tlist:
                    if orient == "xy":
                        S.append(((r, s, t), bid))  # Render in the xy plane
                    elif orient == "yz":
                        S.append(((t, r, s), bid))  # Render in the yz plane
                    elif orient == "xz":
                        S.append(((r, t, s), bid))  # Render in the xz plane
    if not labelme:
        return S
    else:
        return S, L, insts


def disk(radius=5, size=None, bid=DEFAULT_IDM, orient="xy", thickness=1, labelme=False, **kwargs):
    if size is not None:
        radius = max(size // 2, 1)
    assert radius > 0
    N = 2 * radius
    c = N / 2 - 1 / 2
    S = []
    L = {}
    insts = {}
    tlist = range(0, thickness)
    for r in range(N):
        for s in range(N):
            if ((r - c) ** 2 + (s - c) ** 2) ** 0.5 < N / 2:
                for t in tlist:
                    if orient == "xy":
                        S.append(((r, s, t), bid))  # Render in the xy plane
                    elif orient == "yz":
                        S.append(((t, r, s), bid))  # Render in the yz plane
                    elif orient == "xz":
                        S.append(((r, t, s), bid))  # Render in the xz plane

    if not labelme:
        return S
    else:
        return S, L, insts


def rectanguloid(
    size=None, depth=None, height=None, width=None, bid=DEFAULT_IDM, labelme=False, **kwargs
):
    """Construct a solid rectanguloid"""
    if type(size) is int:
        size = [size, size, size]
    if size is None:
        if width is None and height is None and depth is None:
            size = [3, 3, 3]
        else:
            size = [1, 1, 1]
    if width is not None:
        size[0] = width
    if height is not None:
        size[1] = height
    if depth is not None:
        size[2] = depth
    assert size[0] > 0
    assert size[1] > 0
    assert size[2] > 0
    S = []
    L = {}
    for r in range(size[0]):
        for s in range(size[1]):
            for t in range(size[2]):
                S.append(((r, s, t), bid))
    if not labelme:
        return S
    else:
        insts = get_rect_instance_seg((0, size[0] - 1), (0, size[1] - 1), (0, size[2] - 1))
        L = labels_from_instance_seg(insts, L=L)
        return S, L, insts


def near_extremes(x, a, b, r):
    """checks if x is within r of a or b, and between them"""
    assert a <= b
    if x >= a and x <= b:
        if x - a < r or b - x < r:
            return True
    return False


def rectanguloid_frame(
    size=3, thickness=1, bid=DEFAULT_IDM, only_corners=False, labelme=False, **kwargs
):
    """Construct just the lines of a rectanguloid"""
    R = hollow_rectanguloid(size=size, thickness=thickness, bid=bid, labelme=False, **kwargs)
    M = np.max([l for (l, idm) in R], axis=0)
    S = []
    for l, idm in R:
        bne = [near_extremes(l[i], 0, M[i], thickness) for i in range(3)]
        if (only_corners and sum(bne) == 3) or (not only_corners and sum(bne) > 1):
            S.append((l, idm))
    if not labelme:
        return S
    else:
        return S, {}, {}


def hollow_rectanguloid(size=3, thickness=1, bid=DEFAULT_IDM, labelme=False, **kwargs):
    """Construct a rectanguloid that's hollow inside"""
    if type(size) is int:
        size = [size, size, size]
    inner_size = (
        (thickness, size[0] - thickness),
        (thickness, size[1] - thickness),
        (thickness, size[2] - thickness),
    )
    assert size[0] > 0
    assert size[1] > 0
    assert size[2] > 0
    assert inner_size[0][0] > 0 and inner_size[0][1] < size[0]
    assert inner_size[1][0] > 0 and inner_size[1][1] < size[1]
    assert inner_size[2][0] > 0 and inner_size[2][1] < size[2]
    os = size
    rs = inner_size
    S = []
    L = {}
    interior = []
    for r in range(size[0]):
        for s in range(size[1]):
            for t in range(size[2]):
                proceed = False
                proceed = proceed or r < rs[0][0] or r >= rs[0][1]
                proceed = proceed or s < rs[1][0] or s >= rs[1][1]
                proceed = proceed or t < rs[2][0] or t >= rs[2][1]
                if proceed:
                    S.append(((r, s, t), bid))
                else:
                    interior.append((r, s, t))
    if not labelme:
        return S
    else:
        insts = get_rect_instance_seg((0, os[0] - 1), (0, os[1] - 1), (0, os[2] - 1))
        inner_insts = get_rect_instance_seg(
            (rs[0][0] - 1, rs[0][1]), (rs[1][0] - 1, rs[1][1]), (rs[2][0] - 1, rs[2][1])
        )
        L = labels_from_instance_seg(insts, L=L)
        inner_insts = {"inner_" + l: inner_insts[l] for l in inner_insts}
        L = labels_from_instance_seg(inner_insts, L=L)
        insts.update(inner_insts)
        for p in interior:
            L[p] = "inside"
        insts["inside"] = tuple(interior)
        return S, L, insts


def hollow_cube(size=3, thickness=1, bid=DEFAULT_IDM, labelme=False, **kwargs):
    return hollow_rectanguloid(
        size=(size, size, size), thickness=thickness, bid=bid, labelme=labelme
    )


def sphere(radius=5, size=None, bid=DEFAULT_IDM, labelme=False, **kwargs):
    """Construct a solid sphere"""
    if size is not None:
        radius = size // 2
    N = 2 * radius
    c = N / 2 - 1 / 2
    S = []
    L = {}
    insts = {"spherical_surface": [[]]}
    for r in range(N):
        for s in range(N):
            for t in range(N):
                w = ((r - c) ** 2 + (s - c) ** 2 + (t - c) ** 2) ** 0.5
                if w < N / 2:
                    S.append(((r, s, t), bid))
                    if w > N / 2 - 1:
                        if labelme:
                            L[(r, s, t)] = ["spherical_surface"]
                            insts["spherical_surface"][0].append((r, s, t))
    if not labelme:
        return S
    else:
        return S, L, insts


def spherical_shell(radius=5, size=None, thickness=2, bid=DEFAULT_IDM, labelme=False, **kwargs):
    """Construct a sphere that's hollow inside"""
    if size is not None:
        radius = size // 2
    N = 2 * radius
    c = N / 2 - 1 / 2
    S = []
    L = {}
    insts = {"spherical_surface": [[]], "inner_spherical_surface": [[]]}
    for r in range(N):
        for s in range(N):
            for t in range(N):
                in_radius = False
                out_core = False
                w = ((r - c) ** 2 + (s - c) ** 2 + (t - c) ** 2) ** 0.5
                if w < N / 2:
                    in_radius = True
                if w > N / 2 - thickness:
                    out_core = True
                if in_radius and out_core:
                    S.append(((r, s, t), bid))
                    if labelme and w < N / 2 - thickness + 1:
                        L[(r, s, t)] = ["inner_spherical_surface"]
                        insts["inner_spherical_surface"][0].append((r, s, t))
                if labelme and in_radius and not out_core:
                    L[(r, s, t)] = ["inside"]
                if in_radius and labelme and w > N / 2 - 1:
                    L[(r, s, t)] = ["spherical_surface"]
                    insts["spherical_surface"][0].append((r, s, t))

    if not labelme:
        return S
    else:
        return S, L, insts


def square_pyramid(
    slope=1, radius=10, size=None, height=None, bid=DEFAULT_IDM, labelme=False, **kwargs
):
    if size is not None:
        radius = size + 2  # this is a heuristic
    assert slope > 0
    S = []
    L = {}
    insts = {
        "pyramid_peak": [[]],
        "pyramid_bottom_corner": [[], [], [], []],
        "pyramid_bottom_edge": [[], [], [], []],
        "pyramid_diag_edge": [[], [], [], []],
        "pyramid_face": [[], [], [], []],
        "pyramid_bottom": [[]],
    }

    if height is None:
        height = math.ceil(slope * radius)
    for h in range(height):
        start = math.floor(h / slope)
        end = 2 * radius - start
        for s in range(start, end + 1):
            for t in range(start, end + 1):
                S.append(((s, h, t), bid))
                if labelme:
                    sstart = s == start
                    send = s == end
                    tstart = t == start
                    tend = t == end
                    sb = sstart or send
                    tb = tstart or tend
                    if h == height - 1:
                        L[(s, h, t)] = ["pyramid_peak"]
                        insts["pyramid_peak"][0].append((s, h, t))
                    if h == 0:
                        if sstart:
                            insts["pyramid_bottom_edge"][0].append((s, h, t))
                        if send:
                            insts["pyramid_bottom_edge"][1].append((s, h, t))
                        if tstart:
                            insts["pyramid_bottom_edge"][2].append((s, h, t))
                        if tend:
                            insts["pyramid_bottom_edge"][3].append((s, h, t))
                        if sb and tb:
                            L[(s, h, t)] = ["pyramid_bottom_corner"]
                            i = sstart * 1 + tstart * 2
                            insts["pyramid_bottom_corner"][i].append((s, h, t))
                    else:
                        if sstart:
                            insts["pyramid_face"][0].append((s, h, t))
                        if send:
                            insts["pyramid_face"][1].append((s, h, t))
                        if tstart:
                            insts["pyramid_face"][2].append((s, h, t))
                        if tend:
                            insts["pyramid_face"][3].append((s, h, t))
                        if sb and tb:
                            L[(s, h, t)] = ["pyramid_diag_edge"]
                            i = sstart * 1 + tstart * 2
                            insts["pyramid_diag_edge"][i].append((s, h, t))
    if not labelme:
        return S
    else:
        return S, L, insts


def tower(height=10, size=None, base=-1, bid=DEFAULT_IDM, labelme=False, **kwargs):
    if size is not None:
        height = size
    D = height // 3
    if D < 3:
        D = 1
        height = 3
    if base == 1:
        base = 0
    if base <= 0:
        base = -base + 1
        if base > D:
            base = D
        size = (base, height, base)
        return rectanguloid(size=size, bid=bid, labelme=labelme)
    else:
        if base > D:
            base = D
        c = D / 2 - 1 / 2
        S = []
        for s in range(height):
            for m in range(base):
                for n in range(base):
                    if ((m - c) ** 2 + (n - c) ** 2) ** 0.5 <= D / 2:
                        S.append(((m, s, n), bid))
        if not labelme:
            return S
        else:
            return S, {}, {}


def ellipsoid(size=(7, 8, 9), bid=DEFAULT_IDM, labelme=False, **kwargs):
    if type(size) is int:
        size = [size, size, size]
    assert size[0] > 0
    assert size[1] > 0
    assert size[2] > 0

    a = size[0]
    b = size[1]
    c = size[2]

    cx = a - 1 / 2
    cy = b - 1 / 2
    cz = c - 1 / 2

    S = []
    L = {}
    insts = {}
    for r in range(2 * a):
        for s in range(2 * b):
            for t in range(2 * c):
                if (((r - cx) / a) ** 2 + ((s - cy) / b) ** 2 + ((t - cz) / c) ** 2) ** 0.5 < 1.0:
                    S.append(((r, s, t), bid))
    if not labelme:
        return S
    else:
        return S, L, insts


def dome(radius=3, size=None, bid=DEFAULT_IDM, thickness=2, labelme=False, **kwargs):
    """Construct a hemisphere, in the direction of positive y axis"""
    if size is not None:
        radius = max(size // 2, 1)
    assert radius > 0
    N = 2 * radius
    cx = cz = N / 2 - 1 / 2
    cy = 0
    S = []
    L = {}
    insts = {}
    for r in range(N):
        for s in range(N):
            for t in range(N):
                in_radius = False
                out_core = False
                if ((r - cx) ** 2 + (s - cy) ** 2 + (t - cz) ** 2) ** 0.5 < N / 2:
                    in_radius = True
                if ((r - cx) ** 2 + (s - cy) ** 2 + (t - cz) ** 2) ** 0.5 > N / 2 - thickness:
                    out_core = True
                if in_radius and out_core:
                    S.append(((r, s, t), bid))
    if not labelme:
        return S
    else:
        return S, L, insts


def arch(size=3, distance=11, bid=DEFAULT_IDM, orient="xy", labelme=False, **kwargs):
    """Arch is a combination of 2 parallel columns, where the columns
    are connected by a stepped roof.
    Total height of the arch structure:length + distance//2 + 1"""
    length = size  # Length is the height of 2 parallel columns
    # "distance" is the distance between the columns

    L = {}
    insts = {}
    S = []

    assert distance % 2 == 1  # distance between the 2 columns should be odd

    offset = 0
    for i in range(length):
        S.append(((offset, i, offset), bid))
        cx = offset + distance + 1
        cy = i
        cz = offset
        if orient == "xy":
            S.append(((cx, cy, cz), bid))
        elif orient == "yz":
            S.append(((cz, cy, cx), bid))

    # Blocks corresponding to the stepped roof
    for i in range(1, distance // 2 + 1):
        cx_1 = offset + i
        cy_1 = length + i - 1
        cz_1 = offset
        if orient == "xy":
            S.append(((cx_1, cy_1, cz_1), bid))
        elif orient == "yz":
            S.append(((cz_1, cy_1, cx_1), bid))

        cx_2 = offset + distance + 1 - i
        cy_2 = length + i - 1
        cz_2 = offset
        if orient == "xy":
            S.append(((cx_2, cy_2, cz_2), bid))
        elif orient == "yz":
            S.append(((cz_2, cy_2, cx_2), bid))

    # topmost block
    cx_top = offset + distance // 2 + 1
    cy_top = length + distance // 2
    cz_top = offset
    if orient == "xy":
        S.append(((cx_top, cy_top, cz_top), bid))
    elif orient == "yz":
        S.append(((cz_top, cy_top, cx_top), bid))

    if not labelme:
        return S
    else:
        return S, L, insts


def get_rect_instance_seg(bx, by, bz):
    I = {}
    I["top_corner"] = [((bx[i], by[1], bz[j]),) for i in range(2) for j in range(2)]

    I["bottom_corner"] = [((bx[i], by[0], bz[j]),) for i in range(2) for j in range(2)]

    I["vertical_edge"] = [
        tuple((bx[i], s, bz[j]) for s in range(by[0], by[1] + 1))
        for i in range(2)
        for j in range(2)
    ]

    I["top_edge"] = [
        tuple((s, by[1], bz[i]) for s in range(bx[0], bx[1] + 1)) for i in range(2)
    ] + [tuple((bx[i], by[1], s) for s in range(bz[0], bz[1] + 1)) for i in range(2)]

    I["bottom_edge"] = [
        tuple((s, by[0], bz[i]) for s in range(bx[0], bx[1] + 1)) for i in range(2)
    ] + [tuple((bx[i], by[0], s) for s in range(bz[0], bz[1] + 1)) for i in range(2)]

    I["face"] = [
        tuple((s, t, bz[i]) for s in range(bx[0], bx[1] + 1) for t in range(by[0], by[1] + 1))
        for i in range(2)
    ] + [
        tuple((bx[i], t, s) for s in range(bz[0], bz[1] + 1) for t in range(by[0], by[1] + 1))
        for i in range(2)
    ]

    I["top"] = [
        tuple((s, by[1], t) for s in range(bx[0], bx[1] + 1) for t in range(bz[0], bz[1] + 1))
    ]

    I["bottom"] = [
        tuple((s, by[0], t) for s in range(bx[0], bx[1] + 1) for t in range(bz[0], bz[1] + 1))
    ]
    return I


def labels_from_instance_seg(I, L=None):
    L = L or {}
    for label in I:
        for i in I[label]:
            for p in i:
                if L.get(p) is None:
                    L[p] = [label]
                else:
                    if label not in L[p]:
                        L[p].append(label)
    return L


# TODO: merge this with the one in build utils
def get_bounds(S):
    """
    S should be a list of tuples, where each tuple is a pair of
    (x,y,z) and ids
    """
    x, y, z = list(zip(*list(zip(*S))[0]))
    return min(x), max(x), min(y), max(y), min(z), max(z)


# TODO: vector direction?
def mirror(S, axis=0):
    """make a mirror of S"""
    m = get_bounds(S)
    out = []
    aM = m[2 * axis + 1]
    for b in S:
        c = aM - b[0][axis]
        loc = [b[0], b[1], b[2]]
        loc[axis] = c
        out.append(tuple(loc), b[1])

    return out


# NOTE: arrangement will eventually be any of the shapes and
# arrangement will eventually be handled by relpos model
# for now it is either a 'circle' or a 'line'
# schematic is the thing to be built at each location


def cube(size=3, bid=DEFAULT_IDM, labelme=False, **kwargs):
    if type(size) not in (tuple, list):
        size = (size, size, size)
    return rectanguloid(size=size, bid=bid, labelme=labelme)