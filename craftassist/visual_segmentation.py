"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

""" Visualize Segmentation """
import numpy as np
import pickle
import os
import sys
import json
import random
import glob
from tqdm import tqdm


possible_types = [(35, i) for i in range(1, 16)]
possible_types.extend(
    [
        (1, 0),
        (2, 0),
        (3, 0),
        (5, 0),
        (7, 0),
        (11, 0),
        (12, 0),
        (12, 1),
        (13, 0),
        (17, 0),
        (17, 2),
        (18, 0),
        (19, 0),
        (20, 0),
        (21, 0),
        (22, 0),
        (23, 0),
        (24, 0),
        (25, 0),
    ]
)

""" Given components, return a generated numpy file. """


def gen_segmentation_npy(raw_data, name, save_folder="../../house_segment/"):
    seg_data = []
    inf = 100000
    xmin, ymin, zmin = inf, inf, inf
    xmax, ymax, zmax = -inf, -inf, -inf
    for i in range(len(raw_data)):
        bid, meta = possible_types[i % len(possible_types)]
        for xyz in raw_data[i][1]:
            x, y, z = xyz
            seg_data.append((x, y, z, bid, meta))
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            zmin = min(zmin, z)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
            zmax = max(zmax, z)
    dims = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1, 2]
    mat = np.zeros(dims, dtype="uint8")
    for block in seg_data:
        x, y, z, bid, meta = block
        mat[x - xmin, y - ymin, z - zmin, :] = bid, meta
    npy_schematic = np.transpose(mat, (1, 2, 0, 3))
    save_file = save_folder + name
    np.save(save_file, npy_schematic)
    return save_file + ".npy"


def visualize_segmentation(segment_pkl, save_folder="../../house_segment/"):
    name = segment_pkl.split("/")[-1][:-4]
    if not os.path.exists(save_folder + name):
        os.makedirs(save_folder + name)
    raw_data = np.load(segment_pkl)
    if len(raw_data) > len(possible_types):
        print(
            "Warning: # of segments exceed total number of different blocks %d > %d"
            % (len(raw_data), len(possible_types))
        )
    npy_file = gen_segmentation_npy(raw_data, name, save_folder + name + "/")
    cmd = "python render_schematic.py %s --out-dir=%s" % (npy_file, save_folder + name)
    os.system(cmd)


def visualize_segmentation_by_step(segment_pkl, save_folder="../../house_segment/"):
    name = segment_pkl.split("/")[-1][:-4]
    if not os.path.exists(save_folder + name):
        os.makedirs(save_folder + name)
    folder_name = save_folder + name + "/real"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cmd = "python render_schematic.py %s --out-dir=%s" % (
        "../../minecraft_houses/%s/schematic.npy" % name,
        folder_name,
    )
    os.system(cmd)

    raw_data = np.load(segment_pkl)
    if len(raw_data) > len(possible_types):
        print(
            "Warning: # of segments exceed total number of different blocks %d > %d"
            % (len(raw_data), len(possible_types))
        )
    fname = "../../minecraft_houses/%s/placed.json" % name
    build_data = json.load(open(fname, "r"))
    time_data = {}
    inf = 100000
    minx, miny, minz = inf, inf, inf
    maxx, maxy, maxz = -inf, -inf, -inf
    for i, item in enumerate(build_data):
        x, y, z = item[2]
        if item[-1] == "B":
            continue
        assert item[-1] == "P"
        minx = min(minx, x)
        miny = min(miny, y)
        minz = min(minz, z)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
        maxz = max(maxz, z)
    for i, item in enumerate(build_data):
        x, y, z = item[2]
        y -= miny
        z -= minz
        x -= minx
        time_data[(x, y, z)] = i
    segments_time = []
    for i in range(len(raw_data)):
        avg_time = []
        for xyz in raw_data[i][1]:
            try:
                avg_time.append(time_data[xyz])
            except:
                from IPython import embed

                embed()
        avg_time = np.mean(avg_time)
        segments_time.append((avg_time, i))
    segments_time = sorted(segments_time)
    order_data = []
    for avg_time, cid in segments_time:
        order_data.append(raw_data[cid])
    order_file = open("%s" % save_folder + "order_data/%s.pkl" % name, "wb")
    pickle.dump(order_data, order_file)
    order_file.close()

    for i in range(len(order_data)):
        npy_file = gen_segmentation_npy(
            order_data[: i + 1], name + "_step%d" % (i), save_folder + name + "/"
        )
        folder_name = save_folder + name + "/step%d" % i
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        cmd = "python render_schematic.py %s --out-dir=%s" % (npy_file, folder_name)
        os.system(cmd)


def save_order_data(segment_pkl, save_folder="../../house_segment/"):
    name = segment_pkl.split("/")[-1][:-4]
    raw_data = np.load(segment_pkl)
    fname = "../../minecraft_houses/%s/placed.json" % name
    try:
        build_data = json.load(open(fname, "r"))
    except:
        print("can't load %s ... skip" % fname)
        return
    time_data = {}
    inf = 100000
    minx, miny, minz = inf, inf, inf
    maxx, maxy, maxz = -inf, -inf, -inf
    for i, item in enumerate(build_data):
        x, y, z = item[2]
        if item[-1] == "B":
            continue
        assert item[-1] == "P"
        minx = min(minx, x)
        miny = min(miny, y)
        minz = min(minz, z)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
        maxz = max(maxz, z)
    for i, item in enumerate(build_data):
        x, y, z = item[2]
        y -= miny
        z -= minz
        x -= minx
        time_data[(x, y, z)] = i
    segments_time = []
    for i in range(len(raw_data)):
        avg_time = []
        for xyz in raw_data[i][1]:
            try:
                avg_time.append(time_data[xyz])
            except:
                from IPython import embed

                embed()
        avg_time = np.mean(avg_time)
        segments_time.append((avg_time, i))
    segments_time = sorted(segments_time)
    order_data = []
    for avg_time, cid in segments_time:
        order_data.append(raw_data[cid])
    order_file = open("%s" % save_folder + "order_data/%s.pkl" % name, "wb")
    pickle.dump(order_data, order_file)
    order_file.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    print("mode: %s" % mode)
    random.seed(10)
    if mode == "auto":
        segment_files = glob.glob("../../house_components/*.pkl")
        random.shuffle(segment_files)
        for i in range(10):
            visualize_segmentation_by_step(segment_files[i])
    elif mode == "all":
        segment_pkl = sys.argv[2]
        visualize_segmentation(segment_pkl)
    elif mode == "stepbystep":
        segment_pkl = sys.argv[2]
        visualize_segmentation_by_step(segment_pkl)
    elif mode == "saveorder":
        segment_files = glob.glob("../../house_components/*.pkl")
        for i in tqdm(range(len(segment_files)), desc="save order data"):
            save_order_data(segment_files[i])
    else:
        print("mode not found!")
