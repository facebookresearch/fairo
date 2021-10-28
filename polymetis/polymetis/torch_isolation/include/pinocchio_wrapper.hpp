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

C_TORCH_EXPORT struct State *initialize(const std::string &ee_link_name,
                                        const std::string &xml_buffer);

C_TORCH_EXPORT Eigen::VectorXd
get_lower_position_limits(struct State *pinocchio_state);
C_TORCH_EXPORT Eigen::VectorXd
get_upper_position_limits(struct State *pinocchio_state);
C_TORCH_EXPORT Eigen::VectorXd
get_velocity_limits(struct State *pinocchio_state);
C_TORCH_EXPORT int get_nq(struct State *pinocchio_state);

C_TORCH_EXPORT Eigen::VectorXd forward_kinematics(struct State *pinocchio_state,
                                                  const Eigen::VectorXd &q);
C_TORCH_EXPORT void
compute_jacobian(State *state, const Eigen::VectorXd &joint_positions,
                 Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>> &J);
C_TORCH_EXPORT Eigen::Matrix<double, Eigen::Dynamic, 1>
inverse_dynamics(State *state, const Eigen::VectorXd &q,
                 const Eigen::VectorXd &v, const Eigen::VectorXd &a);

C_TORCH_EXPORT void inverse_kinematics(State *state,
                                       const Eigen::Vector3d &ee_pos_,
                                       const Eigen::Quaterniond &ee_quat,
                                       const Eigen::VectorXd rest_pose,
                                       double eps = 1e-4,
                                       int64_t max_iters = 1000,
                                       double dt = 0.1, double damping = 1e-12);
} // namespace pinocchio_wrapper

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif // PINOCCHIO_WRAPPER_OPS_H
