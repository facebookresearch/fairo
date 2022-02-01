import ctypes
import numpy as np
import os

rvl = ctypes.cdll.LoadLibrary(os.path.abspath("rvl.so"))

buf = np.zeros((640*480*2), dtype=np.uint8)
out = np.zeros((640*480*2), dtype=np.uint16)

def rvl_compress(depth):
    # assert depth.dtype == np.uint16
    # ptr = depth.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    # buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
    ptr = ctypes.c_void_p(depth.ctypes.data)
    buf_ptr = ctypes.c_void_p(buf.ctypes.data)
    num_pix = depth.shape[0] * depth.shape[1]
    out_length = rvl.CompressRVL(ptr, buf_ptr, ctypes.c_int(num_pix))
    out = np.copy(buf[0:out_length])
    return [out, depth.shape]

def rvl_decompress(inp):
    comp, shape = inp
    # assert comp.dtype == np.uint8
    ptr = ctypes.c_void_p(comp.ctypes.data)
    out_ptr = ctypes.c_void_p(out.ctypes.data)
    num_pix = shape[0] * shape[1]
    rvl.DecompressRVL(ptr, out_ptr, ctypes.c_int(num_pix))
    out_ = np.copy(out[0:(num_pix)])
    out_ = out_.reshape(shape)
    return out_

if __name__ == "__main__":
    # inp = np.ones((640, 480), dtype=np.uint16)
    inp = np.load("test_pcd_serdes.npy")
    comp = rvl_compress(inp)
    print("Compressed size", comp[0].shape)
    out_ = rvl_decompress(comp)
    print("out shape", out_.shape)

    print('allclose', np.allclose(inp, out_))
