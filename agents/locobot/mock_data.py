import numpy as np
import cv2
import io
import blosc
import math
import zarr
import rvl_lib as rvl
import zipfile
import bz2
import zlib
import pyzstd as zstd_
import pickle
from pyzstd import CParameter

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
        zarr.save('test4.zarr', self.depth)
        self.rgb = cv2.imread("test_pcd_serdes.jpg")

        self.df32 = self.depth.astype(np.float32)
        self.rf32 = self.rgb.astype(np.float32)

        with open("zstd_dict_depth.pkl", "rb") as f:
            self.zstd_dict = zstd_.ZstdDict(pickle.load(f))
        self.zstd_option = {CParameter.nbWorkers : 4, CParameter.compressionLevel : 1}
        #        self.zstd_ = zstd_.ZstdCompressor(level_or_option=option, zstd_dict = zstd_dict) # RichMemZstdCompressor(-1)

    def float(self):
        return self.rf32, self.df32

    def int(self):
        return self.rgb, self.depth

    def psnr(self, img1, img2, range_=255.):
        rpsnr = 0 # compute_psnr(self.rgb, rgb)
        return compute_psnr(img1, img2, range_)

    def webp(self):
        rgb = self.rgb
        quality=80
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        fmt = ".webp"
        _, rgb_data = cv2.imencode(fmt, rgb, encode_param)
        return rgb_data

    def webp_decode(self, rgb):
        return cv2.imdecode(rgb, cv2.IMREAD_COLOR)

    def png(self):
        rgb = self.rgb
        fmt = ".png"
        _, rgb_data = cv2.imencode(fmt, rgb)
        return rgb_data

    def png_decode(self, rgb):
        return cv2.imdecode(rgb, cv2.IMREAD_COLOR)

    def jpg(self):
        rgb = self.rgb
        quality=80
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        fmt = ".jpg"
        _, rgb_data = cv2.imencode(fmt, rgb, encode_param)
        return rgb_data

    def jpg_decode(self, rgb):
        return cv2.imdecode(rgb, cv2.IMREAD_COLOR)

    def npz(self):
        compressed_depth = io.BytesIO()
        np.savez_compressed(compressed_depth, self.depth)
        compressed_depth.seek(0)
        compressed = compressed_depth.read()
        return compressed

    def npz_decode(self, depth):
        depth_out = np.load(io.BytesIO(depth))['arr_0']
        return depth_out

    def bloscx(self, cname='blosclz', clevel=9, shuffle=blosc.NOSHUFFLE):
        depth_out = blosc.pack_array(self.depth, cname=cname, clevel=clevel, shuffle=shuffle)
        return depth_out

    def bloscx_decode(self, depth):
        depth_out = blosc.unpack_array(depth)
        return depth_out

    def bloscrvl(self, cname='blosclz', clevel=9, shuffle=blosc.NOSHUFFLE):
        depth_out = rvl.rvl_compress(self.depth)
        depth_out = blosc.pack_array(depth_out[0], cname=cname, clevel=clevel, shuffle=shuffle)
        return [depth_out, self.depth.shape]

    def bloscrvl_decode(self, depth_):
        depth, shape = depth_
        depth_out = blosc.unpack_array(depth)
        depth_out = rvl.rvl_decompress([depth_out, shape])
        return depth_out

    def rvl(self):
        depth_out = rvl.rvl_compress(self.depth)
        return depth_out

    def rvl_decode(self, depth):
        depth_out = rvl.rvl_decompress(depth)
        return depth_out

    def zip(self):
        shape = self.depth.shape
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False, compresslevel=3) as zip_file:
        # with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_BZIP2, False, compresslevel=2) as zip_file:
            zip_file.writestr("1.txt", self.depth.tobytes())
        zip_buffer.seek(0)
        compressed = zip_buffer.read()
        return [compressed, shape]

    def zip_decode(self, depth_):
        depth, shape = depth_
        z = zipfile.ZipFile(io.BytesIO(depth))
        buf = z.read(z.infolist()[0])
        depth_out = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return depth_out

    def bz2(self):
        shape = self.depth.shape
        compressed = bz2.compress(self.depth.tobytes(), compresslevel=9)
        return [compressed, shape]

    def bz2_decode(self, depth_):
        depth, shape = depth_
        buf = bz2.decompress(depth)
        depth_out = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return depth_out

    def zlib(self):
        shape = self.depth.shape
        compressed = zlib.compress(self.depth.tobytes(), level=3)
        return [compressed, shape]

    def zlib_decode(self, depth_):
        depth, shape = depth_
        buf = zlib.decompress(depth)
        depth_out = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return depth_out

    def bloscptr(self, cname='blosclz', clevel=9, shuffle=blosc.NOSHUFFLE):
        a = self.depth
        data_ptr = a.__array_interface__['data'][0]
        depth_out = blosc.compress_ptr(data_ptr, a.size, a.dtype.itemsize,
                                       clevel=clevel, shuffle=shuffle, cname=cname)
        return [depth_out, self.depth.shape]

    def bloscptr_decode(self, depth_):
        depth, shape = depth_
        a2 = np.empty(shape, dtype=np.uint16)
        blosc.decompress_ptr(depth, a2.__array_interface__['data'][0])
        a2 = a2.reshape(shape)
        return a2

    def zstd(self):
        shape = self.depth.shape
        compressed = zstd_.compress(self.depth.tobytes(), level_or_option=self.zstd_option, zstd_dict=self.zstd_dict)
        return [compressed, shape]

    def zstd_decode(self, depth_):
        depth, shape = depth_
        buf = zstd_.decompress(depth, zstd_dict=self.zstd_dict)
        depth_out = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return depth_out
        
        
        
