"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import colorsys
import random
import numpy as np
import base64
import io
import torch
import torchvision
import webcolors

from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import defaultdict


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


colors = random_colors(20)


def get_random_color():
    return colors[0]


def get_encoded_image(file):
    with open(file, "rb") as image:
        f = base64.b64encode(image.read())
    return f


def get_decoded_image(fstr):
    image = base64.b64decode(fstr)
    return Image.open(io.BytesIO(image)).convert("RGB")


def get_np_decoded_image(enc):
    a = base64.decodebytes(enc)
    b = np.frombuffer(a, dtype=np.uint8)
    print("decoded pros {}, {}".format(type(b), b.shape))
    return Image.fromarray(b, "RGB")


def draw_xyz(img, xyz, cent):
    d = ImageDraw.Draw(img)
    d.text(cent, str(xyz[0]) + ",\n" + str(xyz[1]) + ",\n" + str(xyz[2]), fill=(255, 255, 255))
    d.point(cent, fill=(255, 0, 0))
    return img


def get_closest_color_name(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_color_tag(img, cent):
    x = int(cent[0])
    y = int(cent[1])
    color = get_closest_color_name(img.getpixel((x, y)))
    return color


def get_coords(masked, img, xyz, centers):
    # decode
    # xyz is in row-major, centers corresponds to the xy for the image (which is in column major)
    # xyz = base64.decodebytes(enc_xyz)
    # coords = np.frombuffer(xyz, dtype=np.float64).reshape((4,-1))
    coords = np.around(xyz, decimals=2)
    # print("Decode success ? {} \n{}".format(coords.shape, coords[:, :5]))
    # print("Image size {}".format(img.size))
    # map each x,y to a an index into coords
    # img = draw_xyz(img, [0,0,0], (0,0))
    marker_dict = defaultdict(int)
    id = 4
    for cent in centers:
        xyz = coords[:3, cent[1] * img.size[0] + cent[0]]  # what should this index be
        # xyz = coords[:3, cent[0]*img.size[1] + cent[1]]
        # what should this index be
        # xyz = [xyz[1], xyz[0], xyz[2]]
        marker_dict[id] = {
            "position": [xyz[0], xyz[1], xyz[2]],
            "color": get_closest_color_name(img.getpixel((int(cent[0]), int(cent[1])))),
        }
        masked = draw_xyz(masked, xyz, cent)
        id += 1
    # draw the coords on the img
    return masked, marker_dict
