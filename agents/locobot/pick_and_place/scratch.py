import numpy as np
from utils import get_pcd_in_cam
from home_robot.ros.camera import RosCamera
import rospy

depth = np.load("depth.npy")
rgb = np.load("rgb.npy")
intrinsic_mat = np.array([
    [603.00891113, 0., 323.16085815],
    [0., 601.89398193, 245.71160889],
    [0., 0., 1.]
])

print(depth.shape, depth.min(), depth.max())
print(rgb.shape, rgb.min(), rgb.max())

pcd1 = get_pcd_in_cam(depth, intrinsic_mat)
print(pcd1.shape, pcd1.min(), pcd1.max())

rospy.init_node('debug')
dpt_cam = RosCamera('/camera/aligned_depth_to_color', buffer_size=1)
pcd2 = dpt_cam.depth_to_xyz(dpt_cam.fix_depth(np.rot90(depth, k=1, axes=(0, 1))))
print(pcd2.shape, pcd2.min(), pcd2.max())
