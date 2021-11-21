import numpy as np
import copy


def pcd_ground_seg_pca(scan, th=0.80, z_offset=-1.1):
  """ Perform PCA over PointCloud to segment ground.
  """
  pcd = copy.deepcopy(scan)
  _, covariance = pcd.compute_mean_and_covariance()
  eigen_vectors = np.linalg.eig(covariance)[1]
  k = eigen_vectors.T[2]

  # magnitude of projecting each face normal to the z axis
  normals = np.asarray(scan.normals)
  points = np.asarray(scan.points)
  mag = np.linalg.norm(np.dot(normals, k).reshape(-1, 1), axis=1)
  ground = pcd.select_by_index(np.where((mag >= th) & (points[:, 2] < z_offset))[0])
  rest = pcd.select_by_index(np.where((mag >= th) & (points[:, 2] < z_offset))[0], invert=True)

  # Also remove the faces that are looking downwards
  up_normals = np.asarray(ground.normals)
  orientation = np.dot(up_normals, k)
  ground = ground.select_by_index(np.where(orientation > 0.0)[0])

  ground.paint_uniform_color([1.0, 0.0, 0.0])
  rest.paint_uniform_color([0.0, 0.0, 1.0])
  
  return ground, rest


def pcd_ground_seg_open3d(scan, distance_threshold=0.3, ransac_n=3, num_iterations=100):
  """ Open3D also supports segmententation of geometric primitives from point clouds using RANSAC.
  """
  pcd = copy.deepcopy(scan)

  ground_model, ground_indexes = scan.segment_plane(distance_threshold=distance_threshold,
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
  ground_indexes = np.array(ground_indexes)

  ground = pcd.select_by_index(ground_indexes)
  rest = pcd.select_by_index(ground_indexes, invert=True)

  ground.paint_uniform_color([1.0, 0.0, 0.0])
  rest.paint_uniform_color([0.0, 0.0, 1.0])

  return ground, rest
