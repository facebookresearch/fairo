import numpy as np
import cv2
import io
import blosc
import math
import pickle

def compute_psnr(img1, img2, r=255.):
    img1 = img1.astype(np.float64) / r
    img2 = img2.astype(np.float64) / r
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return -100.
    return 10 * math.log10(1. / mse)

class HelloData(object):
    def __init__(self):
        self.depth = np.load("test_pcd_serdes.npy")
        self.rgb = cv2.imread("test_pcd_serdes.jpg")

    def float(self):
        return self.rf32, self.df32

    def int(self):
        return self.rgb, self.depth

    def psnr(self, img1, img2, range_=255.):
        rpsnr = 0 # compute_psnr(self.rgb, rgb)
        return compute_psnr(img1, img2, range_)

    def jpg(self):
        rgb = self.rgb
        quality=80
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        fmt = ".jpg"
        _, rgb_data = cv2.imencode(fmt, rgb, encode_param)
        return rgb_data

    def jpg_decode(self, rgb):
        return cv2.imdecode(rgb, cv2.IMREAD_COLOR)

    def bloscx(self, cname='zstd', clevel=1, shuffle=blosc.NOSHUFFLE):
        depth_out = blosc.pack_array(self.depth, cname=cname, clevel=clevel, shuffle=shuffle)
        return depth_out

    def bloscx_decode(self, depth):
        depth_out = blosc.unpack_array(depth)
        return depth_out

