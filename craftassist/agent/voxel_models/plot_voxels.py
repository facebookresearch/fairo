"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import plotly.graph_objs as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import visdom
import pickle
import os
import torch

GEOSCORER_DIR = os.path.dirname(os.path.realpath(__file__))
MC_DIR = os.path.join(GEOSCORER_DIR, "../../../")

A = Axes3D  # to make flake happy :(


def draw_color_hash(schematic, vis, title="", threshold=0.05, win=None, bins=3):
    """schematic is DxHxW, each entry an index into hash bin"""
    clrs = []
    schematic = schematic.cpu()
    X = torch.nonzero(schematic)
    clrs = np.zeros((X.shape[0], 3))
    for i in range(X.shape[0]):
        r = schematic[X[i][0], X[i][1], X[i][2]]
        r = r - 1
        clrs[i][2] = r % bins
        r = r - clrs[i][2]
        clrs[i][1] = r / bins % bins
        r = r - clrs[i][1] * bins
        clrs[i][0] = r / bins ** 2
    clrs = (256 * clrs / bins).astype("int64")
    w = vis.scatter(
        X=X.numpy(),
        win=win,
        opts={
            "markercolor": clrs,
            "markersymbol": "square",
            "markersize": 15,
            "title": title,
            "camera": dict(eye=dict(x=2, y=0.1, z=2)),
        },
    )
    vis._send({"win": w, "camera": dict(eye=dict(x=2, y=0.1, z=2))})
    return w


def draw_rgb(schematic, vis, title="", threshold=0.05, win=None, colorio=2):
    """Draw rgb plot of schematics"""
    clrs = []
    schematic = schematic.cpu()
    szs = schematic.shape
    X = torch.nonzero(schematic[:3, :, :, :].norm(2, 0) > threshold)
    U = schematic.view(szs[0], -1).t()
    X_lin = szs[2] * szs[3] * X[:, 0] + szs[3] * X[:, 1] + X[:, 2]
    clrs = U[X_lin]
    clrs = torch.clamp(clrs, 0, 1)
    if clrs.shape[1] == 1:
        clrs = clrs.repeat(1, 3)
        clrs = clrs / 2
    colors = (256 * clrs[:, 0:3]).long().numpy()
    w = vis.scatter(
        X=X,
        win=win,
        opts={
            "markercolor": colors,
            "markersymbol": "square",
            "markersize": 15,
            "title": title,
            "camera": dict(eye=dict(x=2, y=0.1, z=2)),
        },
    )
    vis._send({"win": w, "camera": dict(eye=dict(x=2, y=0.1, z=2))})
    return w


def cuboid_data(pos, size=(1, 1, 1)):
    """code taken from
    https://stackoverflow.com/a/35978146/4124317
    suppose axis direction: x: to left; y: to inside; z: to upper
    get the (left, outside, bottom) point
    """
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]
    y = [
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1], o[1], o[1]],
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]
    z = [
        [o[2], o[2], o[2], o[2], o[2]],
        [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), color=(0, 1, 0, 1), ax=None):
    """Plotting a cube element at position pos"""
    if ax is not None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)


class SchematicPlotter:
    """Schematic Plotter"""

    def __init__(self, viz):
        self.viz = viz
        ims = pickle.load(
            open(os.path.join(MC_DIR, "minecraft_specs/block_images/block_data"), "rb")
        )
        colors = []
        alpha = []
        self.bid_to_index = {}
        self.index_to_color = {}
        self.bid_to_color = {}
        count = 0
        for b, I in ims["bid_to_image"].items():
            I = I.reshape(1024, 4)
            if all(I[:, 3] < 0.2):
                colors = (0, 0, 0)
            else:
                colors = I[I[:, 3] > 0.2, :3].mean(axis=0) / 256.0
            alpha = I[:, 3].mean() / 256.0
            self.bid_to_color[b] = (colors[0], colors[1], colors[2], alpha)
            self.bid_to_index[b] = count
            self.index_to_color[count] = (colors[0], colors[1], colors[2], alpha)
            count = count + 1

    def drawMatplot(self, schematic, n=1, title=""):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_aspect("equal")
        if type(schematic) is np.ndarray:
            for i in range(schematic.shape[0]):
                for j in range(schematic.shape[1]):
                    for k in range(schematic.shape[2]):
                        if schematic[i, j, k, 0] > 0:
                            c = self.bid_to_color.get(tuple(schematic[i, j, k, :]))
                            if c:
                                plotCubeAt(pos=(i, k, j), color=c, ax=ax)  # x, z, y
        else:
            for b in schematic:
                if b[1][0] > 0:
                    c = self.bid_to_color.get(b[1])
                    if c:
                        plotCubeAt(pos=(b[0][0], b[0][2], b[0][1]), color=c, ax=ax)  # x, z, y
        plt.title(title)
        visrotate(n, ax, self.viz)

        return fig, ax

    def drawGeoscorerPlotly(self, schematic):
        x = []
        y = []
        z = []
        id = []
        if type(schematic) is torch.Tensor:
            sizes = list(schematic.size())
            for i in range(sizes[0]):
                for j in range(sizes[1]):
                    for k in range(sizes[2]):
                        if schematic[i, j, k] > 0:
                            x.append(i)
                            y.append(j)
                            z.append(k)
                            id.append(schematic[i, j, k].item())
        elif type(schematic) is np.ndarray:
            for i in range(schematic.shape[0]):
                for j in range(schematic.shape[1]):
                    for k in range(schematic.shape[2]):
                        if schematic[i, j, k, 0] > 0:
                            c = self.bid_to_color.get(tuple(schematic[i, j, k, :]))
                            if c:
                                x.append(i)
                                y.append(j)
                                z.append(k)
                                id.append(i + j + k)
        else:
            for b in schematic:
                if b[1][0] > 0:
                    c = self.bid_to_color.get(b[1])
                    if c:
                        x.append(b[0][0])
                        y.append(b[0][2])
                        z.append(b[0][1])
                        id.append(i + j + k)
        trace1 = go.Scatter3d(
            x=np.asarray(x).transpose(),
            y=np.asarray(y).transpose(),
            z=np.asarray(z).transpose(),
            mode="markers",
            marker=dict(
                size=5,
                symbol="square",
                color=id,
                colorscale="Viridis",
                line=dict(color="rgba(217, 217, 217, 1.0)", width=0),
                opacity=1.0,
            ),
        )
        data = [trace1]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        self.viz.plotlyplot(fig)
        return fig

    def drawPlotly(self, schematic, title="", ptype="scatter"):
        x = []
        y = []
        z = []
        id = []
        clrs = []
        if type(schematic) is torch.Tensor:
            sizes = list(schematic.size())
            for i in range(sizes[0]):
                for j in range(sizes[1]):
                    for k in range(sizes[2]):
                        if schematic[i, j, k] > 0:
                            x.append(i)
                            y.append(j)
                            z.append(k)
                            id.append(schematic[i, j, k].item())
        elif type(schematic) is np.ndarray:
            for i in range(schematic.shape[0]):
                for j in range(schematic.shape[1]):
                    for k in range(schematic.shape[2]):
                        if schematic[i, j, k, 0] > 0:
                            c = self.bid_to_color.get(tuple(schematic[i, j, k, :]))
                            if c:
                                x.append(i)
                                y.append(j)
                                z.append(k)
                                id.append(i + j + k)
                                clrs.append(c)
        else:
            for b in schematic:
                if b[1][0] > 0:
                    c = self.bid_to_color.get(b[1])
                    if c:
                        x.append(b[0][0])
                        y.append(b[0][2])
                        z.append(b[0][1])
                        id.append(i + j + k)
                        clrs.append(c)
        #                        clrs.append(self.bid_to_index[b[1]])
        if ptype == "scatter":
            X = torch.Tensor([x, y, z]).t()
            if len(clrs) == 0:
                raise Exception("all 0 input?")
            colors = (256 * torch.Tensor(clrs)[:, 0:3]).long().numpy()
            w = self.viz.scatter(
                X=X,
                opts={
                    "markercolor": colors,
                    "markersymbol": "square",
                    "markersize": 15,
                    "title": title,
                    "camera": dict(eye=dict(x=2, y=0.1, z=2)),
                },
            )
            #            layout = go.Layout(camera =dict(eye=dict(x=2, y=.1, z=2)))
            self.viz._send({"win": w, "camera": dict(eye=dict(x=2, y=0.1, z=2))})
            return w
        else:
            maxid = max(clrs)
            clr_set = set(clrs)
            cmap = [
                [
                    c / maxid,
                    "rgb({},{},{})".format(
                        self.index_to_color[c][0],
                        self.index_to_color[c][1],
                        self.index_to_color[c][0],
                    ),
                ]
                for c in clr_set
            ]
            trace1 = go.Volume(
                x=np.asarray(x).transpose(),
                y=np.asarray(y).transpose(),
                z=np.asarray(z).transpose(),
                value=np.asarray(clrs).transpose(),
                isomin=0.1,
                isomax=0.8,
                colorscale=cmap,
                opacity=0.1,  # needs to be small to see through all surfaces
                surface_count=21,  # needs to be a large number for good volume rendering
            )
            data = [trace1]
            layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
            fig = go.Figure(data=data, layout=layout)
            self.viz.plotlyplot(fig)
        return fig


def visrotate(n, ax, viz):
    for angle in range(45, 405, 360 // n):
        ax.view_init(30, angle)
        plt.draw()
        viz.matplot(plt)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="shapes",
        help="which\
                        dataset to visualize (shapes|segments)",
    )
    opts = parser.parse_args()

    CRAFTASSIST_DIR = os.path.join(GEOSCORER_DIR, "../")
    sys.path.append(CRAFTASSIST_DIR)

    vis = visdom.Visdom(server="http://localhost")
    sp = SchematicPlotter(vis)
    # fig, ax = sp.drawMatplot(schematic, 4, "yo")

    if opts.dataset == "shapes":
        import shape_dataset as sdata

        num_examples = 3
        num_neg = 3
        dataset = sdata.SegmentCenterShapeData(
            nexamples=num_examples, for_vis=True, useid=True, shift_max=10, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
    elif opts.dataset == "segments":
        import inst_seg_dataset as idata

        num_examples = 1
        num_neg = 1
        dataset = idata.SegmentCenterInstanceData(
            nexamples=num_examples, shift_max=10, nneg=num_neg
        )
        for n in range(num_examples):
            curr_data = dataset[n]
            sp.drawPlotly(curr_data[0])
            for i in range(num_neg):
                sp.drawPlotly(curr_data[i + 1])
    else:
        raise Exception("Unknown dataset: {}".format(opts.dataset))


"""            
            oldc = clrs[0]
            clrs[0] = 0
            maxid = max(clrs)
            clr_set = set(clrs)
            cmap = [[c/maxid, "rgb({},{},{})".format(self.index_to_color[c][0],
                                                     self.index_to_color[c][1],
                                                     self.index_to_color[c][0])]
                    for c in clr_set]
#            clrs[0] = oldc
            trace1 = go.Scatter3d(
                x=np.asarray(x).transpose(),
                y=np.asarray(y).transpose(),
                z=np.asarray(z).transpose(),
                mode="markers",
                marker=dict(
                    size=15,
                    symbol="square",
                    color=clrs,
#                    color=id,
                    colorscale=cmap,
#                    colorscale="Viridis",
                    line=dict(color="rgba(217, 217, 217, 1.0)", width=0),
                    opacity=1.0,
                ),
            )
"""
