"""
Copyright (c) Facebook, Inc. and its affiliates.
"""


def yzx_to_dicts(yzx, y_offset=63):
    """Converts yzx format array into a dictionary
    with keys: x, y, z, id and meta"""
    yzx = yzx["schematic"]
    ysize, zsize, xsize, _ = yzx.shape
    if ysize + y_offset > 255:
        raise ValueError("Shape is too big {}".format(yzx.shape))
    blocks = []
    for y in range(ysize):
        for z in range(zsize):
            for x in range(xsize):
                bid, bmeta = yzx[y, z, x, :]
                blocks.append(
                    {"x": x, "y": y + y_offset, "z": z, "id": int(bid), "meta": int(bmeta)}
                )
    return blocks


def dicts_to_lua(dicts):
    """Convert the dictionary to lua format"""
    block_strs = []
    for d in dicts:
        s = "{" + ",".join("{}".format(d[k]) for k in ["x", "y", "z", "id", "meta"]) + "}"
        block_strs.append(s)
    return "{" + ",".join(block_strs) + "}"
