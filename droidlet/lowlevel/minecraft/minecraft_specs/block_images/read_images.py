"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json
import matplotlib.pyplot
import numpy as np
from scipy.ndimage import imread
import visdom
import pickle

# Update this when running this file
home_dir = ""

f = open(home_dir + "/minecraft_specs/block_images/css_chunk.txt")
r = f.readlines()
l = r[0]
q = l.split(".items-28-")

g = open(home_dir + "/minecraft_specs/block_images/html_chunk.txt")
s = g.readlines()
name_to_bid = {}
bid_to_name = {}
for line in s:
    c = line.find('"id">')
    if c > 0:
        d = line.find("<", c)
        idlist = line[c + 5 : d].split(":")
        if len(idlist) == 1:
            idlist.append("0")
    c = line.find('"name">')
    if c > 0:
        d = line.find("<", c)
        name = line[c + 7 : d].lower()
        bid = (int(idlist[0]), int(idlist[1]))
        name_to_bid[name] = bid
        bid_to_name[bid] = name

bid_to_offsets = {}
for line in q:
    s = line.find("png)")
    if s > 0:
        t = line.find("no-")
        offsets = line[s + 5 : t].replace("px", "").split()
        int_offsets = [int(offsets[0]), int(offsets[1])]
        t = line.find("{")
        ids = line[:t].split("-")
        bid = (int(ids[0]), int(ids[1]))
        bid_to_offsets[bid] = int_offsets

big_image = matplotlib.pyplot.imread(home_dir + "/minecraft_specs/block_images/all_blocks")

bid_to_image = {}
name_to_image = {}

for name in name_to_bid:
    bid = name_to_bid[name]
    offsets = bid_to_offsets[bid]
    small_image = big_image[
        -offsets[1] : -offsets[1] + 32, -offsets[0] : -offsets[0] + 32, :
    ].copy()

    bid_to_image[bid] = small_image
    name_to_image[name] = small_image


out = {
    "bid_to_image": bid_to_image,
    "name_to_image": name_to_image,
    "bid_to_name": bid_to_name,
    "name_to_bid": name_to_bid,
}

f.close()
g.close()

f = open(home_dir + "/minecraft_specs/block_images/block_data", "wb")
pickle.dump(out, f)
f.close()


COLOR_NAMES = [
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "green",
    "gray",
    "lime",
    "maroon",
    "navy",
    "olive",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
    "orange",
    "brown",
    "sienna",
    "pink",
    "light yellow",
    "dark yellow",
    "dark yellow",
    "gold",
    "gold",
]
COLORS = np.array(
    (
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.5),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0, 0.5),
        (1.0, 0.0, 0.0),
        (0.75, 0.75, 0.75),
        (0.0, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.65, 0.0),
        (139 / 255, 69 / 255, 19 / 255),
        (160 / 255, 82 / 255, 45 / 255),
        (255 / 255, 192 / 255, 203 / 255),
        (200 / 255, 200 / 255, 50 / 255),
        (200 / 255, 200 / 255, 50 / 255),
        (255 / 255, 255 / 255, 130 / 255),
        (255 / 255, 215 / 255, 40 / 255),
        (255 / 255, 215 / 255, 0 / 255),
    )
)

COLOR_NORMS = np.linalg.norm(COLORS, axis=1) ** 2


CMAP = {
    "aqua": "blue",
    "black": "black",
    "blue": "blue",
    "fuchsia": "purple",
    "green": "green",
    "gray": "gray",
    "lime": "green",
    "maroon": "red",
    "navy": "blue",
    "olive": "green",
    "purple": "purple",
    "red": "red",
    "pink": "pink",
    "silver": "silver",
    "teal": "blue",
    "white": "white",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "sienna": "brown",
    "gold": "yellow",
    "light yellow": "yellow",
    "dark yellow": "yellow",
}


def get_colors(im):
    cim = im[:, :, :3].astype("float32")
    cim = cim.reshape(1024, 3)
    alpha = im[:, :, 3].astype("float32")
    alpha = alpha.reshape(1024)
    cim /= 255
    alpha /= 255
    dists = np.zeros((1024, COLORS.shape[0]))
    for i in range(1024):
        for j in range(COLORS.shape[0]):
            dists[i, j] = ((COLORS[j] - cim[i]) ** 2).sum()
    idx = dists.argmin(axis=1)
    colors = {}
    for i in range(1024):
        if alpha[i] > 0.2:
            if colors.get(COLOR_NAMES[idx[i]]) is None:
                colors[COLOR_NAMES[idx[i]]] = 1
            else:
                colors[COLOR_NAMES[idx[i]]] += 1
    if alpha.mean() < 0.4:
        colors["translucent"] = True
    return colors


name_to_colors = {}
colors_to_name = {}
name_to_simple_colors = {}
simple_colors_to_name = {}

# for i in name_to_image:
#     c = get_colors(name_to_image[i])
#     for j in c:
#         if c[j] > 100:
#             if name_to_colors.get(i) is None:
#                 name_to_colors[i] = [j]
#             else:
#                 name_to_colors[i].append(j)

#             if name_to_simple_colors.get(i) is None:
#                 name_to_simple_colors[i] = [CMAP[j]]
#             else:
#                 name_to_simple_colors[i].append(CMAP[j])

#             if colors_to_name.get(j) is None:
#                 colors_to_name[j] = [i]
#             else:
#                 colors_to_name[j].append(i)

#             if simple_colors_to_name.get(CMAP[j]) is None:
#                 simple_colors_to_name[CMAP[j]] = [i]
#             else:
#                 simple_colors_to_name[CMAP[j]].append(i)


# out = {'name_to_colors':name_to_colors,
#        'name_to_simple_colors':name_to_simple_colors,
#        'colors_to_name':colors_to_name,
#        'simple_colors_to_name':simple_colors_to_name,
#        'cmap':CMAP}

# f = open(home_dir + '/minecraft_specs/block_images/color_data','wb')
# pickle.dump(out, f)
# f.close()


with open(home_dir + "/minecraft_specs/block_images/block_property_data.json") as f:
    block_property_data = json.load(f)

name_to_properties = {}
properties_to_name = {}

for name in name_to_image:
    if name in block_property_data:
        properties = block_property_data[name]["properties"]
        for property in properties:
            if name_to_properties.get(name) is None:
                name_to_properties[name] = [property]
            else:
                name_to_properties[name].append(property)

            if properties_to_name.get(property) is None:
                properties_to_name[property] = [name]
            else:
                properties_to_name[property].append(name)

out = {"name_to_properties": name_to_properties, "properties_to_name": properties_to_name}

f = open(home_dir + "/minecraft_specs/block_images/block_property_data", "wb")
pickle.dump(out, f)
f.close()

with open(home_dir + "/minecraft_specs/block_images/mob_property_data.json") as f:
    mob_property_data = json.load(f)

name_to_properties = {}
properties_to_name = {}

for name in name_to_image:
    if name in mob_property_data:
        properties = mob_property_data[name]["properties"]
        for property in properties:
            if name_to_properties.get(name) is None:
                name_to_properties[name] = [property]
            else:
                name_to_properties[name].append(property)

            if properties_to_name.get(property) is None:
                properties_to_name[property] = [name]
            else:
                properties_to_name[property].append(name)

out = {"name_to_properties": name_to_properties, "properties_to_name": properties_to_name}

f = open(home_dir + "/minecraft_specs/block_images/mob_property_data", "wb")
pickle.dump(out, f)
f.close()


"""
COLORS = {
    'aqua': np.array((0.0, 1.0, 1.0)),
    'black': np.array((0.0, 0.0, 0.0)),
    'blue': np.array((0.0, 0.0, 1.0)),
    'fuchsia': np.array((1.0, 0.0, 1.0)),
    'green': np.array((0.0, .5, 0.0)),
    'gray': np.array((.5, .5, .5)),
    'lime': np.array((0.0, 1.0, 0.0)),
    'maroon': np.array((.5, 0.0, 0.0)),
    'navy': np.array((0.0, 0.0, .5)),
    'olive': np.array((.5, .5, 0.0)),
    'purple': np.array((.5, 0, .5)),
    'red': np.array((1.0, 0.0, 0.0)),
    'silver': np.array((.75, .75, .75)),
    'teal':  np.array((0.0, .5, .5)),
    'white': np.array((1.0, 1.0, 1.0)),
    'yellow': np.array((1.0, 1.0, 0.0)),
    'orange': np.array((1.0, .65, 0.0)),
    'brown': np.array((139/255  69/255  19/255)),
    'sienna': np.array((160/255,  82/255,  45/255)),
    'pink': np.array((255/255, 192/255, 203/255))
}
"""
