import os
import shutil
import glob
import numpy as np
import cv2
from PIL import Image
import skimage.morphology
from natsort import natsorted
import time
import json

from segmentation.constants import map_color_palette


class ObjectGoalNavigationVisualization:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, goal_name="no goal", path="images/default"):
        self.path = path
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path)
        os.makedirs(f"{self.path}/trajectory")
        os.makedirs(f"{self.path}/planner")

        self.vis_image = np.ones((655, 1005, 3)).astype(np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "Predicted Semantic Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 480 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(
            self.vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
        )

        # draw object goal
        text = "Observations (Goal: {})".format(goal_name)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (480 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(
            self.vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
        )

        # draw outlines
        color = [100, 100, 100]
        self.vis_image[49, 15:495] = color
        self.vis_image[49, 510:990] = color
        self.vis_image[50:530, 14] = color
        self.vis_image[50:530, 495] = color
        self.vis_image[50:530, 509] = color
        self.vis_image[50:530, 990] = color
        self.vis_image[530, 15:495] = color
        self.vis_image[530, 510:990] = color

        # draw legend
        legend = cv2.imread("visualization/legend.png")
        lx, ly, _ = legend.shape
        self.vis_image[537 : 537 + lx, 75 : 75 + ly, :] = legend

        self.snapshot_idx = 1
        self.start_time = time.time()
        self.semantic_map = None
        self.semantic_map_vis = None
        self.goal_map = np.zeros((480, 480))
        self.semantic_frame = None
        self.semantic_frame_vis = None
        self.rgb_frame = None
        self.depth_frame = None
        self.pcd = None
        self.pose = None
        self.all_poses = []
        self.action = None
        self.all_actions = []
        self.collision = None
        self.num_collisions = 0

    def snapshot(self):
        snapshot_path = f"{self.path}/trajectory/step{self.snapshot_idx}"
        os.makedirs(snapshot_path)
        os.makedirs(f"{snapshot_path}/frames")
        os.makedirs(f"{snapshot_path}/maps")

        # Summary vis
        cv2.imwrite(f"{snapshot_path}/summary.png", self.vis_image)

        # Frames
        cv2.imwrite(f"{snapshot_path}/frames/rgb.png", self.rgb_frame)
        cv2.imwrite(
            f"{snapshot_path}/frames/depth.png",
            ((self.depth_frame / self.depth_frame.max()) * 255).astype(np.uint8),
        )
        np.save(f"{snapshot_path}/frames/depth.npy", self.depth_frame.astype(np.float32))
        np.save(f"{snapshot_path}/frames/pcd.npy", self.pcd.astype(np.float32))
        cv2.imwrite(f"{snapshot_path}/frames/semantic.png", self.semantic_frame_vis)
        np.save(f"{snapshot_path}/frames/semantic.npy", self.semantic_frame.astype(np.bool))

        # Maps
        cv2.imwrite(f"{snapshot_path}/maps/semantic_and_goal_map.png", self.semantic_map_vis)
        np.save(f"{snapshot_path}/maps/semantic_map.npy", self.semantic_map.astype(np.float32))
        np.save(f"{snapshot_path}/maps/goal_map.npy", self.goal_map.astype(np.bool))

        # Metrics
        json.dump(
            {
                "timestamp": time.time() - self.start_time,
                "start_pose": self.pose,
                "action": self.action,
                "collision": self.collision,
            },
            open(f"{snapshot_path}/logs.json", "w"),
        )

        self.snapshot_idx += 1

    def record_aggregate_metrics(self, last_pose):
        self.all_poses.append(last_pose)
        path_length = sum(
            [
                np.linalg.norm(np.abs(np.array(end[:2]) - np.array(start[:2])))
                for start, end in zip(self.all_poses[:-1], self.all_poses[1:])
            ]
        )
        json.dump(
            {
                "time": time.time() - self.start_time,
                "path_length": path_length,
                "num_steps": len(self.all_actions),
                "num_collisions": self.num_collisions,
                "poses": self.all_poses,
                "actions": self.all_actions,
            },
            open(f"{self.path}/aggregate_logs.json", "w"),
        )

    # def record_video(self):
    #     img_array = []
    #     for filename in natsorted(glob.glob(f"{self.path}/*.png")):
    #         img = cv2.imread(filename)
    #         height, width, _ = img.shape
    #         size = (width, height)
    #         img_array.append(img)
    #
    #     out = cv2.VideoWriter(f"{self.path}/video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)
    #     for i in range(len(img_array)):
    #         out.write(img_array[i])
    #     out.release()

    def set_location_goal(self, goal_map):
        self.goal_map = goal_map

    def set_action_and_collision(self, logs):
        self.action = logs["action"]
        self.all_actions.append(self.action)
        self.collision = logs["collision"]
        if self.collision:
            self.num_collisions += 1

    def update_last_position_vis_info(self, vis_info):
        self.rgb_frame = vis_info["rgb"]
        self.depth_frame = vis_info["depth"]
        self.pcd = vis_info["pcd"]
        self.semantic_frame = vis_info["semantic_frame"]
        self.update_semantic_frame(vis_info["semantic_frame_vis"])
        self.update_semantic_map(vis_info["semantic_map"])
        self.pose = vis_info["pose"]
        self.all_poses.append(self.pose)

    def update_semantic_frame(self, vis):
        """Visualize first-person semantic segmentation frame."""
        self.semantic_frame_vis = vis
        vis = cv2.resize(vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:495] = vis

    def update_semantic_map(self, sem_map):
        """Visualize top-down semantic map."""
        self.semantic_map = sem_map

        sem_channels = sem_map[4:]
        sem_channels[-1] = 1e-5
        obstacle_mask = np.rint(sem_map[0]) == 1
        explored_mask = np.rint(sem_map[1]) == 1
        visited_mask = sem_map[3] == 1
        sem_map = sem_channels.argmax(0)
        no_category_mask = sem_map == sem_channels.shape[0] - 1

        sem_map += 4
        sem_map[no_category_mask] = 0
        sem_map[np.logical_and(no_category_mask, explored_mask)] = 2
        sem_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
        sem_map[visited_mask] = 3

        selem = skimage.morphology.disk(6)
        goal_map = 1 - skimage.morphology.binary_dilation(self.goal_map, selem) != True
        goal_mask = goal_map == 1
        sem_map[goal_mask] = 3

        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette([int(x * 255.0) for x in map_color_palette])
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.transpose(sem_map_vis, (1, 0, 2))
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)

        self.semantic_map_vis = sem_map_vis
        self.vis_image[50:530, 510:990] = sem_map_vis
