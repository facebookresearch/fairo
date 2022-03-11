import pickle
import logging
from tqdm import tqdm
import torch

from fairotag.camera import CameraModule
from scipy.spatial.transform import Rotation as R


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def h_vector(x_3):
    result = torch.ones(4).double()
    result[:3] = x_3
    return result


def h_matrix(rotation, translation):
    result = torch.eye(4).double()
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


class Problem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p_ee_marker = torch.nn.Parameter(0.1 * torch.randn(3))
        self.T_cam_base_rot = torch.nn.Parameter(0.1 * torch.eye(3))
        self.T_cam_base_trans = torch.nn.Parameter(0.1 * torch.zeros(3))

    def forward(self, T_base_ee, p_cam):
        result = (
            h_vector(self.p_ee_marker)
            + T_base_ee @ h_matrix(self.T_cam_base_rot, self.T_cam_base_trans) @ h_vector(p_cam)
        )[:3]
        return result


def get_detections_from_data(data):
    intrinsics_list = data[0]["intrinsics"]
    camera_modules = [CameraModule() for _ in intrinsics_list]
    for intrinsics, camera_module in zip(intrinsics_list, camera_modules):
        camera_module.set_intrinsics(
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            ppx=intrinsics["ppx"],
            ppy=intrinsics["ppy"],
            coeffs=intrinsics["coeffs"],
        )
        camera_module.register_marker_size(19, 0.02625)  # TODO: verify!

    detections_by_cam = [[] for _ in range(len(camera_modules))]
    for data_dict in data:
        total_detections = 0
        for i, (img, cam) in enumerate(zip(data_dict["imgs"], camera_modules)):
            markers = cam.detect_markers(img)
            pos, ori = data_dict["pos"], data_dict["ori"]
            assert len(markers) <= 1, "Should never be more than 1 marker visible per image"
            if len(markers) == 1:
                total_detections += 1
                detections_by_cam[i].append([markers[0], pos, ori])
    for i, detections in enumerate(detections_by_cam):
        log.info(f"Camera {i} had {len(detections)} detections")

    return camera_modules, detections_by_cam


if __name__ == "__main__":
    filepath = "caldata.pkl"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    camera_modules, detections_by_cam = get_detections_from_data(data)
    for i, (camera_module, detections) in enumerate(zip(camera_modules, detections_by_cam)):
        log.info(f"Solving camera {i}")

        problem = Problem()
        optimizer = torch.optim.Adam(problem.parameters(), lr=1e-6)
        # proj_matrix = torch.DoubleTensor(camera_module._intrinsics2matrix(camera_module.intrinsics))
        # inv_proj_matrix = torch.linalg.inv(proj_matrix)
        n_epochs = 100
        for epoch in tqdm(range(n_epochs)):
            total_loss = 0
            for marker, pos, ori in detections:
                T_base_ee = h_matrix(
                    rotation=torch.DoubleTensor(R.from_quat(ori).as_matrix()),
                    translation=pos.double(),
                )
                # T_marker_cam = inv_proj_matrix

                p_cam = torch.DoubleTensor(marker.pose.translation())
                result = problem.forward(T_base_ee, p_cam)
                loss = result.norm()
                total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {total_loss}")
