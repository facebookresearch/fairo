import numpy as np
from scipy.spatial.transform import Rotation


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1, 3))
    XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
    XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cm, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    n_z_bins = len(z_bins) + 1
    isnotnan = np.logical_not(np.isnan(XYZ_cm[:, 0]))
    X_bin = np.round(XYZ_cm[:, 0] / xy_resolution).astype(np.int32)
    Y_bin = np.round(XYZ_cm[:, 1] / xy_resolution).astype(np.int32)
    Z_bin = np.digitize(XYZ_cm[:, 2], bins=z_bins).astype(np.int32)

    isvalid = np.array(
        [
            X_bin >= 0,
            X_bin < map_size,
            Y_bin >= 0,
            Y_bin < map_size,
            Z_bin >= 0,
            Z_bin < n_z_bins,
            isnotnan,
        ]
    )
    isvalid = np.all(isvalid, axis=0)

    # TODO: Check this part (not understood indexing properly)
    ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
    ind[np.logical_not(isvalid)] = 0
    count = np.bincount(
        ind.ravel(), isvalid.ravel().astype(np.int32), minlength=map_size * map_size * n_z_bins
    )
    counts = np.reshape(count, [map_size, map_size, n_z_bins])

    return counts


def get_relative_state(cur_state, init_state):
    """
    helpful for calculating the relative state of cur_state wrt to init_state [both states are wrt same frame]
    :param cur_state: frame for which position to be calculated
    :param init_state: frame in which position to be calculated
    
    :type cur_state: tuple [x_robot, y_robot, yaw_robot]
    :type init_state: tuple [x_robot, y_robot, yaw_robot]

    :return: relative state of cur_state wrt to init_state
    :rtype list [x_robot_rel, y_robot_rel, yaw_robot_rel]
    """
    # get relative in global frame
    rel_X = cur_state[0] - init_state[0]
    rel_Y = cur_state[1] - init_state[1]
    # transfer from global frame to init frame
    R = np.array(
        [
            [np.cos(init_state[2]), np.sin(init_state[2])],
            [-np.sin(init_state[2]), np.cos(init_state[2])],
        ]
    )
    rel_x, rel_y = np.matmul(R, np.array([rel_X, rel_Y]).reshape(-1, 1))

    return rel_x[0], rel_y[0], cur_state[2] - init_state[2]

