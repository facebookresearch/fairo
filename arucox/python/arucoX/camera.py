from typing import Dict

import copy
import pickle
from collections import namedtuple

import numpy as np
import sophus as sp
import cv2

# Default parameters
GRID_WIDTH = 5
GRID_HEIGHT = 7
GRID_SQUARE_LENGTH = 0.035
GRID_MARKER_LENGTH = 0.02625

# Marker info struct
MarkerInfo = namedtuple("MarkerInfo", "id corner length pose")
CameraIntrinsics = namedtuple("CameraIntrinsics", "fx, fy, ppx, ppy, dist_coeffs")


class CameraModule:
    def __init__(self, dictionary=None, parameters=None, criteria=None):
        # Aruco hyperparams
        if dictionary is None:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.dictionary = dictionary

        if parameters is None:
            parameters = cv2.aruco.DetectorParameters_create()
        self.parameters = parameters

        if criteria is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        self.criteria = criteria

        # Initialize
        self.intrinsics = None
        self.registered_markers = {}

    def register_marker_size(self, marker_id, length):
        """ Enable pose estimation of given marker ID by registering length of marker """
        self.registered_markers[marker_id] = length

    def detect_markers(self, img):
        # Detect markers in img
        corners, ids, rejected_candidates = cv2.aruco.detectMarkers(
            img, dictionary=self.dictionary, parameters=self.parameters
        )

        # Return empty list if no marker found
        if ids is None:
            return []

        # Estimate pose of detected markers
        num_markers = len(corners)
        ids = ids.squeeze(-1)
        if self.intrinsics is None:
            print(
                "Warning: Intrinsics not set in CameraModule. Pose estimation of markers unavailble."
            )

        else:
            poses = []
            lengths = []
            for id, corner in zip(ids, corners):
                # No pose estimation if marker id not registered
                if id not in self.registered_markers:
                    poses.append(None)
                    lengths.append(None)
                    continue

                # Pose estimation using registered length
                length = self.registered_markers[id]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner,
                    length,
                    self._intrinsics2matrix(self.intrinsics),
                    self.intrinsics.dist_coeffs,
                )
                r = rvecs[0].squeeze()
                t = tvecs[0].squeeze()
                pose = sp.SE3(sp.SO3.exp(r).matrix(), t)

                lengths.append(length)
                poses.append(pose)

        # Output
        markers = []
        for id, corner, length, pose in zip(ids, corners, lengths, poses):
            marker = MarkerInfo(id, corner.squeeze(), length, pose)
            markers.append(marker)

        return markers

    def estimate_marker_pose(self, img, marker_id):
        # Check if marker registered
        if marker_id not in self.registered_markers:
            print("Warning: Marker not registered. Unable to estimate marker pose.")
            return None

        # Detect markers
        markers = self.detect_markers(img)

        # Locate target marker and return pose
        for m in markers:
            if m.id == marker_id:
                return m.pose

        # Return none if marker not found
        return None

    def render_markers(self, img, markers=None):
        # Detect markers if not given
        if markers is None:
            markers = self.detect_markers(img)

        # Draw marker edges & ids
        corners = [m.corner[None, ...] for m in markers]
        ids = np.array([m.id for m in markers])
        img_rend = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

        # Draw marker axes
        for m in markers:
            if m.length is not None:
                tvec = m.pose.translation()[None, :]
                rvec = m.pose.so3().log()[None, :]
                length = m.length / 2.0
                img_rend = cv2.aruco.drawAxis(
                    img_rend,
                    self._intrinsics2matrix(self.intrinsics),
                    self.intrinsics.dist_coeffs,
                    rvec,
                    tvec,
                    length,
                )

        return img_rend

    def calibrate_camera(self, imgs, board=None):
        if board is None:
            board = cv2.aruco.CharucoBoard_create(
                GRID_WIDTH, GRID_HEIGHT, GRID_SQUARE_LENGTH, GRID_MARKER_LENGTH, self.dictionary
            )

        # Detect grid board markers in input calibration imgs
        all_corners = []
        all_ids = []
        for img in imgs:
            if len(img.shape) > 2:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            corners, ids, rejected_candidates = cv2.aruco.detectMarkers(
                img_gray, board.dictionary, parameters=self.parameters
            )

            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(
                        img_gray,
                        corner,
                        winSize=(20, 20),
                        zeroZone=(-1, -1),
                        criteria=self.criteria,
                    )
                ret, interp_corners, interp_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, img_gray, board
                )
                if (
                    interp_corners is not None
                    and interp_ids is not None
                    and len(interp_corners) > 3
                ):
                    all_corners.append(interp_corners)
                    all_ids.append(interp_ids)

        # Calibrate camera from marker corners
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_gray.shape, None, None
        )

        self.intrinsics = CameraIntrinsics(
            fx=mtx[0, 0], fy=mtx[1, 1], ppx=mtx[0, 2], ppy=mtx[1, 2], dist_coeffs=dist.squeeze()
        )

    def get_intrinsics(self) -> CameraIntrinsics:
        return self.intrinsics

    def set_intrinsics(
        self, intrinsics=None, matrix=None, dist_coeffs=np.zeros(5), **kwargs
    ) -> None:
        if intrinsics is not None:
            assert type(intrinsics) is CameraIntrinsics
            self.intrinsics = intrinsics
            return

        if matrix is not None:
            fx = matrix[0, 0]
            fy = matrix[1, 1]
            ppx = matrix[0, 2]
            ppy = matrix[1, 2]

        else:
            fx = kwargs["fx"]
            fy = kwargs["fy"]
            ppx = kwargs["ppx"]
            ppy = kwargs["ppy"]

        self.intrinsics = CameraIntrinsics(fx, fy, ppx, ppy, dist_coeffs)

    @staticmethod
    def _intrinsics2matrix(intrinsics):
        matrix = np.eye(3)
        matrix[0, 0] = intrinsics.fx
        matrix[1, 1] = intrinsics.fy
        matrix[0, 2] = intrinsics.ppx
        matrix[1, 2] = intrinsics.ppy

        return matrix
