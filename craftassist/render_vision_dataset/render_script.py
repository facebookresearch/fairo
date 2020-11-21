"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

#!/usr/bin/python

import os
import sys

if __name__ == "__main__":
    npy_files = sys.argv[1]
    port = int(sys.argv[2])
    home = os.path.expanduser("~")
    ## for each house, we render four different angles

    with open(npy_files, "r") as f:
        lines = f.read().splitlines()

    for l in lines:
        os.system(
            "python render_schematic_with_pixel2block.py %s --out-dir=%s/minecraft/python/stack_agent_this/vision_training/render_results1/new_npys --port=%d"
            % (l, home, port + 25565)
        )
        ## clean up the bin files
#        os.system("rm -f %s/render_results/*bin" % home)
