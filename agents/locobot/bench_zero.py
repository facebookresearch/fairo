import time
import io
import argparse
import blosc
import numpy as np
import timeit
from mock_data import HelloData
from rich import print
from rich.console import Console
from rich.table import Table
import pandas as pd
hello_data = HelloData()

parser = argparse.ArgumentParser(description="Pass in server device IP")
parser.add_argument(
    "--ip",
    help="Server device IP. Default is 192.168.0.0",
    type=str,
    default="0.0.0.0",
)

args = parser.parse_args()

import itertools
import zerorpc
import msgpack
import msgpack_numpy as m
m.patch()

count = 20

console = Console()
if args.ip == 'serdes':
    remote = hello_data
else:
    remote = zerorpc.Client()
    remote.connect("tcp://" + args.ip + ":4242")

def bench(attr, opts=None):
    if attr == 'bloscx' or attr == 'bloscptr' or attr == 'bloscrvl':
        img = getattr(remote, attr)(*opts)
    else:
        img = getattr(remote, attr)()
    if isinstance(img, list):
        size = len(img[0])
    else:
        size = len(img)
    img_out = getattr(hello_data, attr + '_decode')(img)
    return img_out, size

def format_time(tm, count):
    ms = int(1000 * (tm / count))
    if ms == 0:
        fps = 'inf'
    else:
        fps = round(1000 / ms, 1)
    return ms, fps

tables = {}
columns = ["codec", "options", "time (ms)", "fps", "compression", "C-Size (KB)", "O-Size (KB)"]

def log(title, codec, opts, tm, count, img1, img2, size):
    ms, fps = format_time(tm, count)
    original_size = hello_data.rgb.size * hello_data.rgb.dtype.itemsize
    compression = round((original_size) / size, 1)
    compressed_size_kb = int(original_size / 1024)
    original_size_kb = int(size / 1024)
    if not title in tables:
        tables[title] = []
    
    table = tables[title]
    table.append([codec, opts, ms, fps, compression, original_size_kb, compressed_size_kb])

def print_log(title, rank="time (ms)"):
    table = tables[title]
    tb = pd.DataFrame(table, columns=columns)
    tb = tb.sort_values(by=[rank])
    table = tb.values.tolist()
    rt = Table(*columns, title=title)
    for row in table:
        rt.add_row(*[str(r) for r in row])
    print(rt)

# for attr in ['jpg', 'png', 'webp']:
#     tm = timeit.timeit(lambda: bench(attr), number=count)
#     rgb, size = bench(attr)
#     psnr= hello_data.psnr(rgb, hello_data.rgb)
#     print("RGB PSNR", psnr)
#     log("RGB-8", attr, "", tm, count, rgb, hello_data.rgb, size)
    
# print_log("RGB-8")
# for attr in ['int', 'float', 'npz', 'bloscx']:
# for attr in ['zstd', 'zlib', 'bz2', 'zip', 'npz', 'rvl', 'bloscx']: #  'bloscrvl']: # 'bloscptr', 
for attr in ['zstd', 'bloscx']: #  'bloscrvl']: # 'bloscptr', 
    if attr == 'bloscx' or attr == 'bloscptr' or attr == 'bloscrvl':
        # for opts in itertools.product(['lz4', 'lz4hc', 'zlib', 'zstd'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        for opts in itertools.product(['zlib', 'zstd'], [1, 2, 3, 4, 5, 6], [blosc.NOSHUFFLE]):
            tm = timeit.timeit(lambda: bench(attr, opts), number=count)
            depth, size = bench(attr, opts)
            dpsnr = hello_data.psnr(hello_data.depth, depth, range_=65536.)
            assert(dpsnr == -100)
            log("Depth-16", attr, opts, tm, count, depth, hello_data.depth, size)
    else:
        tm = timeit.timeit(lambda: bench(attr), number=count)
        ms, fps = format_time(tm, count)
        depth, size = bench(attr)
        dpsnr = hello_data.psnr(hello_data.depth, depth, range_=65536.)
        assert(dpsnr == -100)
        log("Depth-16", attr, "", tm, count, depth, hello_data.depth, size)


print_log("Depth-16")
