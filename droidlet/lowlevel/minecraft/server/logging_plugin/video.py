"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from matplotlib import pyplot as plt
import glob
import matplotlib.animation as animation
import numpy as np
import os.path
import time

FFMpegWriter = animation.writers["ffmpeg"]


def render_video(ob_dir, outfile, dpi=100, max_depth=48):
    writer = FFMpegWriter(fps=20)

    fig = plt.figure()

    with writer.saving(fig, outfile, dpi):
        t_start = time.time()
        i = 0

        while True:
            fig.clear()

            blockfile = os.path.join(ob_dir, "block.{:08}.bin".format(i))
            if not os.path.isfile(blockfile):
                return

            block = np.fromfile(blockfile, np.uint8).reshape(128, 128)
            plt.subplot(1, 2, 1)
            plt.imshow(block, cmap="prism", animated=True)

            depthfile = os.path.join(ob_dir, "depth.{:08}.bin".format(i))
            depth = np.fromfile(depthfile, np.float32).reshape(128, 128)
            depth[depth > max_depth] = max_depth
            plt.subplot(1, 2, 2)
            plt.imshow(depth, cmap="Blues_r", animated=True)
            plt.title("tick={}".format(i))

            writer.grab_frame()

            i += 1
            avg_fps = i / (time.time() - t_start)
            print("Wrote tick={}, avg_fps={}".format(i, avg_fps))


if __name__ == "__main__":
    import argparse
    import tempfile
    import subprocess
    from recover_initial_blockmap import recover_initial_blockmap

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Directory containing logging.bin")
    parser.add_argument("--outfile", required=True, help="Path to video file to create")
    parser.add_argument(
        "--player-name", required=True, help='Name of player whose eyes to "see" through'
    )
    args = parser.parse_args()
    repo_home = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")

    # Using the seed/config, recover the block map as it was at the start of
    # the world, including all chunks ever loaded by any player
    initial_blockmap_dir = recover_initial_blockmap(args.logdir)

    # Step through each tick, rendering the player's observations to a tempdir
    ob_dir = tempfile.mkdtemp()
    print("Writing observations to:", ob_dir)
    subprocess.check_call(
        [
            os.path.join(repo_home, "bin/log_render"),
            "--out-dir",
            ob_dir,
            "--log-file",
            os.path.join(args.logdir, "logging.bin"),
            "--name",
            args.player_name,
            "--mca-files",
            *glob.glob(os.path.join(initial_blockmap_dir, "*.mca")),
        ]
    )
    print("Wrote observations to:", ob_dir)

    # Render the video from the raw observations
    render_video(ob_dir, args.outfile)
    print("Wrote video to:", args.outfile)
