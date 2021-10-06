import numpy as np
import sophus as sp

from utils.fake_camera_env import FakeCameraEnv

from arucoX.scene import FactorGraph, SceneViz

N_SAMPLES = 10
N_CAMS = 3
MISDETECT_PROB = 0.1


if __name__ == "__main__":
    # Initialize
    scene = SceneViz()
    env = FakeCameraEnv(n_cams=N_CAMS, ws_height=1.0, deviation=0.1, noise=0.02)

    # Add camera poses
    cam_poses = env.get_cam_poses()
    for cam_pose in cam_poses:
        scene.draw_camera(cam_pose)

    # Sample origin marker
    cam_transforms_origin = env.sample_marker(pose=sp.SE3())
    cam_transforms_origin[1] = None
    scene.draw_marker(sp.SE3(), 0.05, color="r")

    # Sample markers
    cam_transforms_ls = []
    for _ in range(N_SAMPLES):
        pose, cam_transforms = env.sample_marker(return_pose=True, misdetect_prob=MISDETECT_PROB)
        scene.draw_marker(pose, 0.05)
        cam_transforms_ls.append(cam_transforms)

    # Optimize
    graph = FactorGraph(N_CAMS)

    for i in range(N_SAMPLES):
        graph.add_marker(cam_transforms_ls[i])
    origin_idx = graph.add_marker(cam_transforms_origin)
    graph.add_marker_prior(origin_idx, sp.SE3(), definite=True)

    result = graph.optimize(verbosity=1)

    # Update scene & visualize
    for c_pose in result["cameras"]:
        scene.draw_camera(c_pose, color="c", axes=False)

    scene.show()
