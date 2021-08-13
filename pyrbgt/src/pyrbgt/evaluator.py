from rbgt_pybind import (
    Body,
    RegionModality,
    RendererGeometry,
    Camera,
    NormalImageRenderer,
    NormalImageViewer,
    filesystem,
    Transform3fA,
    mat2t3fA,
    t3fA2mat,
    Model,
)

from .tracker import TestTracker

import os
import math
import time
import numpy as np
import json
import cv2


class Evaluator(object):
    def __init__(
        self, dataset_path, body_names, sequence_name, save_path, image_handle
    ):
        self.dataset_path = dataset_path
        self.body_names = body_names
        self.sequence_name = sequence_name
        self.save_path = save_path

        os.makedirs(self.save_path, exist_ok=True)

        self.kNFrames = 1001

        self.translation_error_threshold = 0.05
        self.rotation_error_threshold = 5 * math.pi / 180
        self.visualize_all_results = True
        self.sphere_radius = 0.8
        self.n_divides = 4
        self.n_points = 200

        poses_first_file = open(os.path.join(dataset_path, "poses_first.txt"), "r")
        poses_first_file.readline()

        self.poses_first = [Transform3fA() for _ in range(self.kNFrames)]
        for i in range(len(self.poses_first)):
            line = poses_first_file.readline()
            line_list = [float(n.split("\n")[0]) for n in line.split("\t")]
            line_list[9] = line_list[9] / 1000
            line_list[10] = line_list[10] / 1000
            line_list[11] = line_list[11] / 1000
            mat2t3fA(line_list, self.poses_first[i])

        poses_second_file = open(os.path.join(dataset_path, "poses_second.txt"), "r")
        poses_second_file.readline()

        self.poses_second = [Transform3fA() for _ in range(self.kNFrames)]
        for i in range(len(self.poses_second)):
            line = poses_second_file.readline()
            line_list = [float(n.split("\n")[0]) for n in line.split("\t")]
            line_list[9] = line_list[9] / 1000
            line_list[10] = line_list[10] / 1000
            line_list[11] = line_list[11] / 1000
            mat2t3fA(line_list, self.poses_second[i])

        self.renderer_geometry = RendererGeometry()
        self.camera = Camera()
        self.camera.set_name("rbot_camera")
        self.image_handle_initializer = image_handle
        self.image_handle = self.image_handle_initializer(
            self.dataset_path, self.body_names[0], self.sequence_name
        )
        self.tracker = TestTracker(self.image_handle)
        intrinsics = self.image_handle.get_intrinsics()
        self.camera.set_intrinsics(
            intrinsics.fu,
            intrinsics.fv,
            intrinsics.ppu,
            intrinsics.ppv,
            intrinsics.width,
            intrinsics.height,
        )
        self.camera.UpdateImage2(self.image_handle.get_image())
        self.viewer = NormalImageViewer()
        self.viewer.Init("viewer", self.renderer_geometry, self.camera)
        self.tracker.AddViewer(self.viewer)

        self.body = Body(
            "body", filesystem("path_placeholder"), 0.001, True, False, 0.3
        )
        self.model = Model("model")
        self.model.set_use_random_seed(False)
        self.regional_modality = RegionModality()
        self.regional_modality.Init(
            "regional_modality", self.body, self.model, self.camera
        )

        self.tracker.AddRegionModality(self.regional_modality)

        self.tracker.SetUpObjects()

    def Evaluate(self):
        # 0 - frame index, 1 - translation error, 2 - rotation error, 3 - tracking loss,
        # 4 - complete cycle, 5 - calculate before camera update,
        # 6 - calculate correspondences, 7 - calculate pose update
        results = np.zeros((len(self.body_names), self.kNFrames - 1, 9))
        # Iterate over all bodies
        for i_body in range(len(self.body_names)):
            self.InitBodies(self.body_names[i_body])
            self.ResetBody(0)
            # Iterate over all frames
            for i_frame in range(self.kNFrames - 1):
                result = results[i_body][i_frame]
                result[0] = i_frame
                result[4:] = self.ExecuteMeasuredTrackingCycle(i_frame)
                result[1:4] = self.CalculatePoseResults(
                    self.body.body2world_pose(), self.poses_first[i_frame + 1]
                )

                self.ResetBody(i_frame + 1)

            single_body_result = results[i_body]
            tracking_loss = (
                single_body_result[:, 3].sum() / single_body_result[:, 3].size
            )
            translation_error = (
                single_body_result[:, 1].sum() / single_body_result[:, 1].size
            )
            rotation_error = (
                single_body_result[:, 2].sum() / single_body_result[:, 2].size
            )
            complete_cycle = (
                single_body_result[:, 4].sum() / single_body_result[:, 4].size
            )
            calculate_before_camera_update = (
                single_body_result[:, 5].sum() / single_body_result[:, 5].size
            )
            calculate_correspondences = (
                single_body_result[:, 6].sum() / single_body_result[:, 6].size
            )
            calculate_pose_update = (
                single_body_result[:, 7].sum() / single_body_result[:, 7].size
            )

            single_body_path = os.path.join(
                self.save_path, "result_{}.txt".format(self.body_names[i_body])
            )
            single_body_dict = {
                "body_name": self.body_names[i_body],
                "success_rate": 1 - tracking_loss,
                "translation_error": translation_error,
                "rotation_error": rotation_error * 180 / math.pi,
                "complete_cycle": complete_cycle,
                "calculate_before_camera_update": calculate_before_camera_update,
                "calculate_correspondences": calculate_correspondences,
                "calculate_pose_update": calculate_pose_update,
            }
            print(single_body_dict)
            with open(single_body_path, "w") as outfile:
                json.dump(single_body_dict, outfile, indent=2)

    def CalculatePoseResults(self, pose, pose_gt):
        errors = np.zeros(3)
        matrix = np.array(t3fA2mat(pose)).reshape((4, 4))
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        matrix_gt = np.array(t3fA2mat(pose_gt)).reshape((4, 4))
        rotation_gt = matrix_gt[:3, :3]
        translation_gt = matrix_gt[:3, 3]
        translation_error = np.linalg.norm(translation - translation_gt)
        rotation_error = math.acos(
            np.clip((np.matmul(rotation.T, rotation_gt).trace() - 1) / 2, -1, 1)
        )
        tracking_loss = int(
            translation_error > self.translation_error_threshold
            or rotation_error > self.rotation_error_threshold
        )
        errors[0] = translation_error
        errors[1] = rotation_error
        errors[2] = tracking_loss
        return errors

    def ExecuteMeasuredTrackingCycle(self, iteration):
        timer = np.zeros(5)
        start_time = time.time()
        self.tracker.CalculateBeforeCameraUpdate()
        timer[1] = time.time() - start_time
        self.tracker.UpdateCameras()

        for i in range(self.tracker.n_corr_iterations):
            start_time = time.time()
            self.tracker.CalculateCorrespondences(i)
            timer[2] += time.time() - start_time

            # Visualize correspondences
            corr_save_idx = iteration * self.tracker.n_corr_iterations + i

            for j in range(self.tracker.n_update_iterations):
                start_time = time.time()
                self.tracker.CalculatePoseUpdate()
                timer[3] += time.time() - start_time
                update_save_idx = corr_save_idx * self.tracker.n_update_iterations + j

        if self.visualize_all_results:
            self.tracker.UpdateViewers(iteration)
            cv2.imwrite("test.png", self.viewer.normal_image())

        timer[0] = timer[1] + timer[2] + timer[3]
        return timer

    def InitBodies(self, body_name):
        self.image_handle = self.image_handle_initializer(
            self.dataset_path, body_name, self.sequence_name
        )
        self.tracker.image_handle = self.image_handle
        self.camera.UpdateImage2(self.image_handle.get_image())
        self.body.set_name(body_name)
        self.body.set_geometry_path(
            filesystem(
                os.path.join(self.dataset_path, body_name, "{}.obj".format(body_name))
            )
        )
        model_directory = filesystem(os.path.join(self.dataset_path, body_name))
        model_name = body_name + "_model"

        if not self.model.LoadModel(model_directory, model_name):
            self.model.GenerateModel(
                self.body, self.sphere_radius, self.n_divides, self.n_points
            )
            self.model.SaveModel(model_directory, model_name)
        self.renderer_geometry.ClearBodies()
        self.renderer_geometry.AddBody(self.body)

    def ResetBody(self, i_frame):
        self.body.set_body2world_pose(self.poses_first[i_frame])
        self.regional_modality.StartModality()

    def set_translation_error_threshold(self, translation_error_threshold):
        self.translation_error_threshold = translation_error_threshold

    def set_rotation_error_threshold(self, rotation_error_threshold):
        self.rotation_error_threshold = rotation_error_threshold

    def set_visualize_all_results(self, visualize_all_results):
        self.visualize_all_results = visualize_all_results

    def set_sphere_radius(self, sphere_radius):
        self.sphere_radius = sphere_radius

    def set_n_divides(self, n_divides):
        self.n_divides = n_divides

    def set_n_points(self, n_points):
        self.n_points = n_points
