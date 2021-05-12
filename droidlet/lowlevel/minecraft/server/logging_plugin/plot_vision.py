"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
from scipy.misc import imread

INF_DEPTH = 100


def plot(blockpath, plt, imgpath=None, depthpath=None, vis=None, out_path=None, size=None):
    block = np.fromfile(blockpath, np.uint8)

    if size is None:
        width = height = int((len(block) / 2) ** 0.5)
    else:
        width, height = size

    try:
        block = block.reshape((height, width, 2))
    except ValueError:
        print('\nReshape failed. Try using "--size width height"')
        import sys

        sys.exit(1)

    if depthpath is not None:
        depth = np.fromfile(depthpath, np.float32)
        depth = depth.reshape((height, width))
        depth[depth > INF_DEPTH] = INF_DEPTH
    else:
        depth = np.zeros((height, width, "float"))

    if imgpath is not None:
        img = imread(imgpath)
    else:
        img = np.zeros((height, width), "float")

    plt.close()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title(imgpath)
    plt.subplot(2, 2, 3)
    plt.imshow(block[:, :, 0], cmap="prism")
    center = block[30:-30, 30:-30, 0]
    max_, min_ = center.max(), center.min()
    plt.title("block_id range: %d, %d" % (min_, max_))

    plt.subplot(2, 2, 4)
    plt.imshow(depth, cmap="Blues_r")
    center = depth[50:-50, 50:-50]
    max_, min_ = center.max(), center.min()
    plt.title("depth range: %f, %f" % (min_, max_))
    if vis is None:
        if out_path:
            plt.savefig(out_path)
        else:
            plt.show()

    else:
        vis.matplot(plt)
    return block, depth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", required=True, help="e.g. path/to/block.135.bin")
    parser.add_argument("--img", help="e.g. path/to/img.png")
    parser.add_argument("--depth", help="e.g. path/to/depth.135.bin")
    parser.add_argument(
        "--visdom", action="store_true", help="visdom if specified, else matplotlib"
    )
    parser.add_argument("--out_path", help="Output path for image")
    parser.add_argument(
        "--size", type=int, nargs="+", help="width and height, e.g. --size 800 600"
    )
    args = parser.parse_args()

    import matplotlib

    if args.visdom:
        import visdom

        matplotlib.use("Agg")
        vis = visdom.Visdom(server="http://localhost")
    else:
        matplotlib.use("TkAgg")
        vis = None

    import matplotlib.pyplot as plt

    block, depth = plot(
        args.blocks,
        plt,
        imgpath=args.img,
        depthpath=args.depth,
        vis=vis,
        out_path=args.out_path,
        size=args.size,
    )
