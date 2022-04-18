import cv2
import numpy as np
import blosc as bl


def jpg_encode(rgb):
    quality = 80
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    fmt = ".jpg"
    _, rgb_data = cv2.imencode(fmt, rgb, encode_param)
    return rgb_data


def jpg_decode(rgb):
    return cv2.imdecode(rgb, cv2.IMREAD_COLOR)


def blosc_encode(depth, cname="zstd", clevel=1, shuffle=bl.NOSHUFFLE):
    depth_out = bl.pack_array(depth, cname=cname, clevel=clevel, shuffle=shuffle)
    return depth_out


def blosc_decode(depth):
    depth_out = bl.unpack_array(depth)
    return depth_out
