// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef PINOCCHIO_WRAPPER_OPS_H
#define PINOCCHIO_WRAPPER_OPS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define C_TORCH_EXPORT __attribute__((visibility("default")))

namespace pinocchio_wrapper {

struct State;

C_TORCH_EXPORT struct State *initialize(const char *xml_buffer);

C_TORCH_EXPORT void destroy(State *state);

C_TORCH_EXPORT Eigen::VectorXd
get_lower_position_limits(struct State *pinocchio_state);
C_TORCH_EXPORT Eigen::VectorXd
get_upper_position_limits(struct State *pinocchio_state);
C_TORCH_EXPORT Eigen::VectorXd
get_velocity_limits(struct State *pinocchio_state);
C_TORCH_EXPORT int get_nq(struct State *pinocchio_state);

C_TORCH_EXPORT Eigen::VectorXd forward_kinematics(struct State *pinocchio_state,
                                                  const Eigen::VectorXd &q,
                                                  int64_t frame_idx);
C_TORCH_EXPORT void
compute_jacobian(State *state, const Eigen::VectorXd &joint_positions,
                 Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>> &J,
                 int64_t frame_idx);
C_TORCH_EXPORT Eigen::Matrix<double, Eigen::Dynamic, 1>
inverse_dynamics(State *state, const Eigen::VectorXd &q,
                 const Eigen::VectorXd &v, const Eigen::VectorXd &a);

C_TORCH_EXPORT void
inverse_kinematics(State *state, const Eigen::Vector3d &link_pos,
                   const Eigen::Quaterniond &link_quat, int64_t frame_idx,
                   Eigen::VectorXd &rest_pose, double eps = 1e-4,
                   int64_t max_iters = 1000, double dt = 0.1,
                   double damping = 1e-12);
C_TORCH_EXPORT int64_t get_link_idx_from_name(State *state,
                                              const char *link_name);
C_TORCH_EXPORT char *get_link_name_from_idx(State *state, int64_t link_idx);
} // namespace pinocchio_wrapper

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif // PINOCCHIO_WRAPPER_OPS_H
