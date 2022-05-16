import os
import glob
import numpy as np
import cv2
from PIL import Image
import skimage.morphology
from natsort import natsorted

from segmentation.constants import map_color_palette, frame_color_palette


class ObjectGoalNavigationVisualization:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, goal_name, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        self.vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "Predicted Semantic Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(
            self.vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
        )

        # draw object goal
        text = "Observations (Goal: {})".format(goal_name)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (640 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(
            self.vis_image, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA
        )

        # draw outlines
        color = [100, 100, 100]
        self.vis_image[49, 15:655] = color
        self.vis_image[49, 670:1150] = color
        self.vis_image[50:530, 14] = color
        self.vis_image[50:530, 655] = color
        self.vis_image[50:530, 669] = color
        self.vis_image[50:530, 1150] = color
        self.vis_image[530, 15:655] = color
        self.vis_image[530, 670:1150] = color

        # draw legend
        legend = cv2.imread("visualization/legend.png")
        lx, ly, _ = legend.shape
        self.vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend

        self.snapshot_idx = 1
        self.goal_map = np.zeros((480, 480))

    def snapshot(self):
        cv2.imwrite(f"{self.path}/snapshot_{self.snapshot_idx}.png", self.vis_image)
        self.snapshot_idx += 1

    def record_video(self):
        img_array = []
        for filename in natsorted(glob.glob(f"{self.path}/*.png")):
            height, width, _ = img.shape
            size = (width, height)
            img = cv2.imread(filename)
            img_array.append(img)

        out = cv2.VideoWriter(f"{self.path}/video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def add_location_goal(self, goal_map):
        self.goal_map = np.stack((self.goal_map, goal_map), 0).max(0)

    def update_semantic_frame(self, vis):
        """Visualize first-person semantic segmentation frame."""
        vis = vis[:, :, [2, 1, 0]]
        vis = cv2.resize(vis, (640, 480), interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = vis

    def update_semantic_map(self, sem_map):
        """Visualize top-down semantic map."""
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
        # TODO Why does the agent height projection include all the explored area?
        # sem_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
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
        self.vis_image[50:530, 670:1150] = sem_map_vis
