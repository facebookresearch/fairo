import cv2
import numpy as np

from .constants import coco_categories


class PickAndPlaceTask:
    def __init__(self, mover):
        self.nav = mover.nav    # Point goal nav + semantic exploration
        self.slam = mover.slam  # Semantic and obstacle map + last frame
        self.bot = mover.bot    # Main robot class

    def pick_and_place(self, start_receptacle: str, object: str, end_receptacle: str):
        """
        End-to-end pick and place with semantic exploration and mobile
        object grasping and placing.

        Arguments:
            start_receptacle: category of receptacle the object is initially on
            object: category of object to grasp
            end_receptacle: category of receptacle the object ends up on
        """
        print(f"Starting pick {object} from {start_receptacle} and place it "
              f"on {end_receptacle}")
        assert start_receptacle in [
            "chair", "couch", "bed", "toilet", "dining-table", "sink"]
        assert end_receptacle in [
            "chair", "couch", "bed", "toilet", "dining-table", "sink"]
        assert object in ["cup", "bottle"]

        # We would use the navigation service for semantic exploration like below
        self.nav.go_to_object(
            object_goal=start_receptacle,
            episode_id=f"go_to_{start_receptacle}",
            exploration_method="learned",
            debug=False,
            visualize=True,
            max_steps=400,
            start_with_panorama=True,
        )
    
    def pick(self, object: str):
        """
        Mobile grasping of an object category present in the last frame.
        
        Arguments:
            object: category of object to grasp
        """
        print(f"Starting pick {object}")
        assert object in ["cup", "bottle"]
        category_id = coco_categories[object]

        print("Here is the data you have available about the last time the "
              "semantic map got updated, which might be slightly stale if the "
              "robot has been moving:")
        
        info = self.slam.get_last_position_vis_info()
        flat_pcd = info["pcd"]
        flat_object_mask = info["semantic_frame"][:, category_id]
        image_object_mask = info["unfiltered_semantic_frame"][:, :, category_id]
        semantic_frame = info["semantic_frame_vis"]
        obstacle_map = info["semantic_map"][0]
        object_map = info["semantic_map"][4 + category_id]

        print(list(info.keys()))
        print("flat_pcd.shape", flat_pcd.shape)
        print("flat_object_mask.shape", flat_object_mask.shape)
        print("image_object_mask.shape", image_object_mask.shape)
        print("obstacle_map.shape", obstacle_map.shape)
        print("object_map.shape", object_map.shape)
        print()

        cv2.imwrite("semantic_frame.png", semantic_frame)
        cv2.imwrite("image_object_mask.png", (image_object_mask * 255).astype(np.uint8))
        cv2.imwrite("obstacle_map.png", (obstacle_map * 255).astype(np.uint8))
        cv2.imwrite("object_map.png", (object_map * 255).astype(np.uint8))

        print("Here is how to transform robot coordinates to map coordinates:")

        pose_of_last_map_update = info["pose"]
        pose_in_map_coordinates = self.slam.robot2map(
            pose_of_last_map_update[:2])

        print("pose_of_last_map_update", pose_of_last_map_update)
        print("curr_pose_in_map_coordinates", pose_in_map_coordinates)
        print()

    def place(self):
        """Mobile placing of the object picked up."""
        print("Starting place")
