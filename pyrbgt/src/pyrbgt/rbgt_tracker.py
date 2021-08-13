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
    mat2t3fAfullMat,
    t3fA2mat,
    Model,
    OcclusionMaskRenderer,
)

from .tracker import TestTracker
from .manual_initializer import ManualInitializer

import threading

import os
import math
import time
import numpy as np
import json
import copy
import cv2


class RBGTTracker(object):
    def __init__(self, configs):
        self.initialized = False

        self.configs = configs
        self.visualize = self.configs.visualize
        self.evaluate = self.configs.evaluate
        self.threadLock = threading.Lock()
        self.output_lst = []

        self.backend_tracker_thread = None
        self.image_handle = None

    def track(self, image_handle):
        self.image_handle = image_handle
        self.backend_tracker_thread = subThreadTracker(
            self.image_handle,
            self.configs,
            self.output_lst,
            self.threadLock,
            self.visualize,
            self.evaluate,
        )
        self.backend_tracker_thread.start()

    def output(self):
        self.threadLock.acquire()
        return_val = copy.deepcopy(self.output_lst)
        self.threadLock.release()
        return return_val


class subThreadTracker(threading.Thread):
    def __init__(
        self,
        image_handle,
        configs,
        output_lst,
        threadLock,
        visualize=True,
        evaluate=True,
        initializer=ManualInitializer,
    ):
        threading.Thread.__init__(self)
        self.configs = configs
        self.model_configs = self.configs.models
        self.output_lst = output_lst
        self.visualize = visualize
        self.threadLock = threadLock
        self.image_handle = image_handle

        self.evaluate = evaluate

        if self.evaluate:
            self.kNFrames = 1001

            poses_first_file = open(
                os.path.join(self.configs.evaluate_dataset_path, "poses_first.txt"), "r"
            )
            poses_first_file.readline()

            self.poses_first = [Transform3fA() for _ in range(self.kNFrames)]
            for i in range(len(self.poses_first)):
                line = poses_first_file.readline()
                line_list = [float(n.split("\n")[0]) for n in line.split("\t")]
                line_list[9] = line_list[9] / 1000
                line_list[10] = line_list[10] / 1000
                line_list[11] = line_list[11] / 1000
                mat2t3fA(line_list, self.poses_first[i])

            poses_second_file = open(
                os.path.join(self.configs.evaluate_dataset_path, "poses_second.txt"),
                "r",
            )
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
        self.camera.set_name("camera")
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
        image = self.image_handle.get_image()
        self.camera.UpdateImage2(image)

        self.viewer = NormalImageViewer()
        self.viewer.Init("viewer", self.renderer_geometry, self.camera)
        self.tracker.AddViewer(self.viewer)

        self.bodies = {}

        self.occlusion_mask_renderer = OcclusionMaskRenderer()
        self.occlusion_mask_renderer.Init(
            "occlusion_mask_renderer",
            self.renderer_geometry,
            self.camera.world2camera_pose(),
            self.camera.intrinsics(),
            0.01,
            5.0,
        )

        occlusion_mask_id = 1

        for model_config in self.model_configs:
            unit_in_meter = 1.0
            if hasattr(model_config, "unit_in_meter"):
                unit_in_meter = model_config.unit_in_meter
            body = Body(
                model_config.name,
                filesystem(
                    os.path.join(model_config.path, model_config.model_filename)
                ),
                unit_in_meter,
                True,
                True,
                0.5,
            )
            body.set_occlusion_mask_id(occlusion_mask_id)
            occlusion_mask_id += 1
            model = Model(model_config.name + "_model")
            model.set_use_random_seed(False)
            regional_modality = RegionModality()
            regional_modality.Init(
                model_config.name + "_regional_modality", body, model, self.camera
            )
            regional_modality.UseOcclusionHandling(self.occlusion_mask_renderer)

            self.tracker.AddRegionModality(regional_modality)

            if not model.LoadModel(
                filesystem(model_config.path), model_config.name + "_model"
            ):
                model.GenerateModel(body, 0.8, 4, 200)
                model.SaveModel(
                    filesystem(model_config.path), model_config.name + "_model"
                )

            self.bodies[model_config.model] = body

        self.tracker.SetUpObjects()

        self.renderer_geometry.ClearBodies()
        for key, body in self.bodies.items():
            self.renderer_geometry.AddBody(body)

        init_configs = []
        unit_in_meters = []
        for model_config in self.model_configs:
            if (
                not hasattr(model_config, "original_pose")
                or not model_config.original_pose
            ):
                init_configs.append(model_config)
                unit_in_meters.append(
                    self.bodies[model_config.model].geometry_unit_in_meter()
                )
            else:
                t = Transform3fA()
                mat2t3fAfullMat(model_config.original_pose, t)
                self.bodies[model_config.model].set_body2world_pose(t)

        self.initializer = initializer(init_configs, intrinsics, unit_in_meters)
        pose = self.initializer.get_pose(image)

        for name in pose.keys():
            R = pose[name]["rotation"].reshape((3, 3))
            T = pose[name]["translation"].reshape(3)
            R = R.flatten().tolist()
            T = T.flatten().tolist()
            t = Transform3fA()
            mat2t3fA(R + T, t)
            self.bodies[name].set_body2world_pose(t)

    def run(self):

        if self.evaluate:
            self.poses_first_error = []
            self.poses_second_error = []
            self.ResetBody(0)
        i_frame = 0

        for region_modality in self.tracker.region_modalities:
            region_modality.StartModality()
        # Iterate over all frames
        while True:
            if not self.ExecuteMeasuredTrackingCycle(i_frame):
                return

            if self.evaluate:
                for key, body in self.bodies.items():
                    if key == "squirrel_small":
                        self.poses_second_error.append(
                            self.CalculatePoseResults(
                                body.body2world_pose(), self.poses_second[i_frame + 1]
                            )
                        )
                    else:
                        self.poses_first_error.append(
                            self.CalculatePoseResults(
                                body.body2world_pose(), self.poses_first[i_frame + 1]
                            )
                        )
                self.ResetBody(i_frame + 1)

                if i_frame >= 999:
                    result = np.mean(self.poses_first_error, axis=0)
                    print(
                        {
                            "translation_loss": result[0],
                            "rotation_loss": result[1],
                            "success_rate": 1 - result[2],
                        }
                    )
                    break

            for key, body in self.bodies.items():
                body2world_pose = np.array(t3fA2mat(body.body2world_pose())).reshape(
                    (4, 4)
                )
                self.output_lst.append(
                    {
                        "name": key,
                        "translation": body2world_pose[:3, 3],
                        "rotation": body2world_pose[:3, :3],
                    }
                )

            for region_modality in self.tracker.region_modalities:
                region_modality.StartModality()
            i_frame += 1

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
            translation_error > 0.05 or rotation_error > 5 * math.pi / 180
        )
        errors[0] = translation_error
        errors[1] = rotation_error
        errors[2] = tracking_loss
        return errors

    def ExecuteMeasuredTrackingCycle(self, iteration):
        self.tracker.CalculateBeforeCameraUpdate()
        if not self.tracker.UpdateCameras():
            return False

        for i in range(self.tracker.n_corr_iterations):
            self.tracker.StartOcclusionMaskRendering()
            self.tracker.CalculateCorrespondences(i)

            for j in range(self.tracker.n_update_iterations):
                self.tracker.CalculatePoseUpdate()

        if self.visualize:
            self.tracker.UpdateViewers(iteration)
        return True

    def ResetBody(self, i_frame):
        for key, body in self.bodies.items():
            if key == "squirrel_small":
                body.set_body2world_pose(self.poses_second[i_frame])
            else:
                body.set_body2world_pose(self.poses_first[i_frame])
