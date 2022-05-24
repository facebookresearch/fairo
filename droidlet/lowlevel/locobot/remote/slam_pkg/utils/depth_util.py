import numpy as np
import torch
import itertools
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


def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]
    Returns:
        grid: B X nF X W X H X D X ..
    """
    wts_dim = []
    pos_dim = []
    grid_dims = init_grid.shape[2:]

    B = init_grid.shape[0]
    F = init_grid.shape[1]

    n_dims = len(grid_dims)

    grid_flat = init_grid.view(B, F, -1)
    for d in range(n_dims):
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
        pos_d = []
        wts_d = []

        for ix in [0, 1]:
            pos_ix = torch.floor(pos) + ix
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)

            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix
            pos_ix = pos_ix * safe_ix

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    l_ix = [[0, 1] for d in range(n_dims)]
    for ix_d in itertools.product(*l_ix):
        wts = torch.ones_like(wts_dim[0][0])
        index = torch.zeros_like(wts_dim[0][0])

        for d in range(n_dims):
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]
            wts = wts * wts_dim[d][ix_d[d]]

        index = index.long()
        grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)

    grid_flat = torch.round(grid_flat)
    return grid_flat.view(init_grid.shape)
