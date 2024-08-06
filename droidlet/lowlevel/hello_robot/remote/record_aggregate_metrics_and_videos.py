import os
import glob
import cv2
from natsort import natsorted
import numpy as np
import json
import sys


def record_aggregate_metrics_and_videos(trajectory_root_path, video_root_path, method):
    step_log_filenames = natsorted(glob.glob(f"{trajectory_root_path}/trajectory/step*/logs.json"))
    timestamps = []
    for filename in step_log_filenames:
        logs = json.load(open(filename, "r"))
        timestamps.append(logs["timestamp"])
    if not os.path.exists(video_root_path):
        os.makedirs(video_root_path)
    for frame in ["rgb", "depth", "semantic"]:
        print(f"Recording {frame} video matching real time")
        record_video(
            natsorted(glob.glob(f"{trajectory_root_path}/trajectory/step*/frames/{frame}.png")),
            timestamps,
            f"{video_root_path}/{frame}_frame.mp4",
            realtime=True,
        )
    if method == "modular":
        print(f"Recording map video matching real time")
        record_video(
            natsorted(
                glob.glob(
                    f"{trajectory_root_path}/trajectory/step*/maps/semantic_and_goal_map.png"
                )
            ),
            timestamps,
            f"{video_root_path}/semantic_and_goal_map.mp4",
            realtime=True,
        )
        print(f"Recording quick summary video")
        record_video(
            natsorted(glob.glob(f"{trajectory_root_path}/trajectory/step*/summary.png")),
            timestamps,
            f"{video_root_path}/summary.mp4",
            realtime=False,
        )
    elif method == "end_to_end":
        print(f"Recording quick semantic frame summary video")
        record_video(
            natsorted(glob.glob(f"{trajectory_root_path}/trajectory/step*/frames/semantic.png")),
            timestamps,
            f"{video_root_path}/summary.mp4",
            realtime=False,
        )
    if not os.path.exists(f"{trajectory_root_path}/aggregate_logs.json"):
        record_aggregate_metrics(step_log_filenames, f"{trajectory_root_path}/aggregate_logs.json")


def record_aggregate_metrics(step_log_filenames, aggregate_log_filename):
    all_poses = []
    all_actions = []
    num_collisions = 0
    for filename in step_log_filenames:
        logs = json.load(open(filename, "r"))
        all_poses.append(logs["start_pose"])
        all_actions.append(logs["action"])
        if logs["collision"]:
            num_collisions += 1
        time = logs["timestamp"]
    path_length = sum(
        [
            np.linalg.norm(np.abs(np.array(end[:2]) - np.array(start[:2])))
            for start, end in zip(all_poses[:-1], all_poses[1:])
        ]
    )
    json.dump(
        {
            "time": time,
            "path_length": path_length,
            "num_steps": len(all_actions),
            "num_collisions": num_collisions,
            "poses": all_poses,
            "actions": all_actions,
        },
        open(aggregate_log_filename, "w"),
    )


def record_video(image_filenames, image_timestamps, video_filename, fps=30, realtime=True):
    images = []
    for filename in image_filenames:
        image = cv2.imread(filename)
        height, width, _ = image.shape
        size = (width, height)
        images.append(image)
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if realtime:
        prev_timestamp = 0
        for timestamp, image in zip(image_timestamps, images):
            frame_repeats = round((timestamp - prev_timestamp) * fps)
            for _ in range(frame_repeats):
                out.write(image)
            prev_timestamp = timestamp
    else:
        for image in images:
            out.write(image)
    out.release()


if __name__ == "__main__":
    method = sys.argv[1]
    assert method in ["modular", "end_to_end"]

    if method == "modular":
        trajectory_root_paths = [
            *glob.glob("trajectories/*/modular_learned"),
            *glob.glob("trajectories/*/modular_frontier"),
        ]
    else:
        trajectory_root_paths = glob.glob(
            "../../../../../fairo/agents/locobot/trajectories/*/end_to_end"
        )

    for trajectory_root_path in trajectory_root_paths:
        video_root_path = trajectory_root_path.replace("trajectories", "videos")
        print(f"Processing {trajectory_root_path}")
        record_aggregate_metrics_and_videos(trajectory_root_path, video_root_path, method)

    if method == "modular":
        aggregate_log_root_paths = [
            *glob.glob("trajectories/*/modular_learned/*.json"),
            *glob.glob("trajectories/*/modular_frontier/*.json"),
        ]
    else:
        aggregate_log_root_paths = glob.glob(
            "../../../../../fairo/agents/locobot/trajectories/*/end_to_end/*.json"
        )

    for f in aggregate_log_root_paths:
        stats = json.load(open(f, "rb"))
        print(f)
        print(stats["path_length"])
        print(stats["time"])
        print(stats["num_steps"])
        print(stats["num_collisions"])
        print()
