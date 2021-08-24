from rbgt_pybind import (
    Body,
    Camera,
    RegionModality,
    RenderDataBody,
    RendererGeometry,
    # Tracker,
    Viewer,
    filesystem,
    Transform3fA,
    mat2t3fA,
    t3fA2mat,
)

import os
import math
import time
import numpy as np

import cv2


class TestTracker(object):
    def __init__(self, image_handle):
        self.region_modalities = []
        self.viewers = []
        self.cameras = {}
        self.occlusion_renderers = {}

        self.n_corr_iterations = 7
        self.n_update_iterations = 2
        self.visualization_time = 0
        self.viewer_time = 1

        self.start_tracking = False
        self.tracking_started = False

        self.image_handle = image_handle

        self.image_cvMat = None

    def AddRegionModality(self, region_modality):
        self.region_modalities.append(region_modality)

    def AddViewer(self, viewer):
        self.viewers.append(viewer)

    def set_n_corr_iterations(self, n_corr_iterations):
        self.n_corr_iterations = n_corr_iterations

    def set_n_update_iterations(self, set_n_update_iterations):
        self.set_n_update_iterations = set_n_update_iterations

    def set_visualization_time(self, set_visualization_time):
        self.set_visualization_time = set_visualization_time

    def set_viewer_time(self, set_viewer_time):
        self.set_viewer_time = set_viewer_time

    def SetUpObjects(self):
        self.cameras = {}
        self.occlusion_renderers = {}

        for region_modality in self.region_modalities:
            if region_modality.camera_ptr():
                if region_modality.camera_ptr().name() not in self.cameras.keys():
                    self.cameras[
                        region_modality.camera_ptr().name()
                    ] = region_modality.camera_ptr()
            if region_modality.occlusion_mask_renderer_ptr():
                if (
                    region_modality.occlusion_mask_renderer_ptr().name()
                    not in self.occlusion_renderers.keys()
                ):
                    self.occlusion_renderers[
                        region_modality.occlusion_mask_renderer_ptr().name()
                    ] = region_modality.occlusion_mask_renderer_ptr()

        for viewer in self.viewers:
            if viewer.camera_ptr():
                if viewer.camera_ptr().name() not in self.cameras.keys():
                    self.cameras[viewer.camera_ptr().name()] = viewer.camera_ptr()

    def CalculateBeforeCameraUpdate(self):
        for region_modality in self.region_modalities:
            if not region_modality.CalculateBeforeCameraUpdate():
                return False
        return True

    def UpdateCameras(self):
        self.image_cvMat = self.image_handle.get_image()
        if self.image_cvMat is None:
            return False
        for camera in self.cameras.values():
            if not camera.UpdateImage2(self.image_cvMat):
                return False
        return True

    def StartOcclusionMaskRendering(self):
        for occlusion_renderer in self.occlusion_renderers.values():
            if not occlusion_renderer.StartRendering():
                return False
        return True

    def CalculateCorrespondences(self, corr_iteration):
        for region_modality in self.region_modalities:
            if not region_modality.CalculateCorrespondences(corr_iteration):
                return False
        return True

    def CalculatePoseUpdate(self):
        for region_modality in self.region_modalities:
            if not region_modality.CalculatePoseUpdate():
                return False
        return True

    def UpdateViewers(self, save_idx):
        if self.viewers:
            for viewer in self.viewers:
                viewer.UpdateViewer(save_idx)
        return True

    def StartTracker(self, start_tracking):
        self.start_tracking = start_tracking
        self.SetUpObjects()

        i = 0
        while True:
            if self.start_tracking:
                for region_modality in self.region_modalities:
                    if not region_modality.StartModality():
                        return
                self.tracking_started = True
                self.start_tracking = False

            if self.tracking_started:
                if not self.ExecuteTrackingCycle(i):
                    return
            else:
                if not self.ExecuteViewingCycle(i):
                    return

    def ExecuteTrackingCycle(self, i):
        if not self.CalculateBeforeCameraUpdate():
            return False
        if not self.UpdateCameras():
            return False
        for j in range(self.n_corr_iterations):
            corr_save_idx = i * self.n_corr_iterations + j
            if not self.StartOcclusionMaskRendering():
                return False
            if not self.CalculateCorrespondences(j):
                return False
            for update_iter in range(self.n_update_iterations):
                update_save_idx = corr_save_idx * self.n_update_iterations + update_iter
                if not self.CalculatePoseUpdate():
                    return False
        if not self.UpdateViewers(i):
            return False
        return True

    def ExecuteViewingCycle(self, i):
        if not self.UpdateCameras():
            return False
        return self.UpdateViewers(i)
