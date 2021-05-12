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
