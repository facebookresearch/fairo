// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio_wrapper.hpp"

extern "C" {

namespace pinocchio_wrapper {

struct State {
  pinocchio::Model model;
  pinocchio::Data model_data;
  pinocchio::FrameIndex ee_frame_idx;

  Eigen::VectorXd ik_sol_v;
  pinocchio::Data::Matrix6x ik_sol_J;
};

struct State *initialize(const std::string &ee_link_name,
                         const std::string &xml_buffer) {
  pinocchio::Model model;
  pinocchio::urdf::buildModelFromXML(xml_buffer, model);

  struct State *state = new struct State;

  state->model = model;
  state->model_data = pinocchio::Data(model);
  state->ee_frame_idx = model.getBodyId(ee_link_name);
  state->ik_sol_v = Eigen::VectorXd(model.nv);
  state->ik_sol_J = pinocchio::Data::Matrix6x(6, model.nv);

  return state;
}

void destroy(struct State *state) { delete state; }

Eigen::VectorXd get_lower_position_limits(struct State *pinocchio_state) {
  return pinocchio_state->model.lowerPositionLimit;
}
Eigen::VectorXd get_upper_position_limits(struct State *pinocchio_state) {
  return pinocchio_state->model.upperPositionLimit;
}
Eigen::VectorXd get_velocity_limits(struct State *pinocchio_state) {
  return pinocchio_state->model.velocityLimit;
}
int get_nq(struct State *pinocchio_state) { return pinocchio_state->model.nq; }

Eigen::VectorXd forward_kinematics(struct State *pinocchio_state,
                                   const Eigen::VectorXd &q) {
  auto model = pinocchio_state->model;
  auto model_data = pinocchio_state->model_data;
  auto ee_frame_idx = pinocchio_state->ee_frame_idx;

  std::cout << "a" << std::endl;
  pinocchio::forwardKinematics(model, model_data, q);
  std::cout << "b" << std::endl;
  pinocchio::updateFramePlacement(model, model_data, ee_frame_idx);
  std::cout << "c" << std::endl;

  auto pos_data = model_data.oMf[ee_frame_idx].translation().transpose();
  std::cout << "d" << std::endl;
  auto quat_data = Eigen::Quaterniond(model_data.oMf[ee_frame_idx].rotation());
  std::cout << "e" << std::endl;

  Eigen::VectorXd result(7);
  for (int i = 0; i < 3; i++) {
    result[i] = pos_data[i];
  }
  std::cout << "f" << std::endl;
  result[3] = quat_data.x();
  result[4] = quat_data.y();
  result[5] = quat_data.z();
  result[6] = quat_data.w();

  std::cout << "g" << std::endl;
  return result;
}

void compute_jacobian(
    State *state, const Eigen::VectorXd &joint_positions,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>> &J) {
                               std::cout << "1" << std::endl;
  auto model = state->model;
                               std::cout << "2" << std::endl;
  auto model_data = state->model_data;
                               std::cout << "3" << std::endl;
  auto ee_frame_idx = state->ee_frame_idx;
                               std::cout << "4" << std::endl;

  pinocchio::computeFrameJacobian(model, model_data, joint_positions,
                                  ee_frame_idx, pinocchio::LOCAL_WORLD_ALIGNED,
                                  J);
                               std::cout << "5" << std::endl;
}

Eigen::Matrix<double, Eigen::Dynamic, 1>
inverse_dynamics(State *state, const Eigen::VectorXd &q,
                 const Eigen::VectorXd &v, const Eigen::VectorXd &a) {
  auto model = state->model;
  auto model_data = state->model_data;
  auto ee_frame_idx = state->ee_frame_idx;
  return pinocchio::rnea(model, model_data, q, v, a);
}

void inverse_kinematics(State *state, const Eigen::Vector3d &ee_pos_,
                        const Eigen::Quaterniond &ee_quat_,
                        Eigen::VectorXd &ik_sol_p_, double eps,
                        int64_t max_iters, double dt, double damping) {
  auto model_ = state->model;
  auto model_data_ = state->model_data;
  auto ee_frame_idx_ = state->ee_frame_idx;
  auto ik_sol_v_ = state->ik_sol_v;
  auto ik_sol_J_ = state->ik_sol_J;

  auto ee_orient_ = ee_quat_.toRotationMatrix();

  // Initialize IK variables
  const pinocchio::SE3 desired_ee(ee_orient_, ee_pos_);

  ik_sol_J_.setZero();

  Eigen::Matrix<double, 6, 1> err;
  ik_sol_v_.setZero();

  // Reset robot pose
  pinocchio::forwardKinematics(model_, model_data_, ik_sol_p_);
  pinocchio::updateFramePlacement(model_, model_data_, ee_frame_idx_);

  // Solve IK iteratively
  for (int i = 0; i < max_iters; i++) {
    // Compute forward kinematics error
    pinocchio::forwardKinematics(model_, model_data_, ik_sol_p_);
    pinocchio::updateFramePlacement(model_, model_data_, ee_frame_idx_);
    const pinocchio::SE3 dMf =
        desired_ee.actInv(model_data_.oMf[ee_frame_idx_]);
    err = pinocchio::log6(dMf).toVector();

    // Check termination
    if (err.norm() < eps) {
      std::cout << "Ending IK at " << i + 1 << "/" << max_iters << " iteration."
                << std::endl;
      break;
    }

    // Descent solution
    pinocchio::computeFrameJacobian(model_, model_data_, ik_sol_p_,
                                    ee_frame_idx_, pinocchio::LOCAL, ik_sol_J_);

    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = ik_sol_J_ * ik_sol_J_.transpose();
    JJt.diagonal().array() += damping;
    ik_sol_v_.noalias() = -ik_sol_J_.transpose() * JJt.ldlt().solve(err);
    ik_sol_p_ = pinocchio::integrate(model_, ik_sol_p_, ik_sol_v_ * dt);
  }

  if (err.norm() >= eps) {
    std::cerr << "WARNING: IK did not converge!" << std::endl;
  }
}

} // namespace pinocchio_wrapper

} /* extern "C" */
