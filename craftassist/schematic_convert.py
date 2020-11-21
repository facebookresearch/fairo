"""
Copyright (c) Facebook, Inc. and its affiliates.

Tool to convert betweeen .schematic and .npy formats

The output is a .npy formatted 4d-array of uint8 with shape (y, z, x, 2), where
the last dimension is id/meta.
"""
import argparse
import os
import subprocess
import tempfile
import gzip

from repo import repo_home

BIN = os.path.join(repo_home, "bin", "schematic_convert")

parser = argparse.ArgumentParser()
parser.add_argument("schematic", help=".schematic file to read")
parser.add_argument("out_file", help="File to write the .npy format to")
args = parser.parse_args()

if not os.path.isfile(BIN):
    subprocess.check_call("make schematic_convert", shell=True, cwd=repo_home)

# read the schematic file
with open(args.schematic, "rb") as f:
    bs = gzip.GzipFile(fileobj=f).read()

unzipped = tempfile.mktemp()
with open(unzipped, "wb") as f:
    f.write(bs)

subprocess.check_call([BIN, unzipped, args.out_file])
