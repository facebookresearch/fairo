// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "spdlog/spdlog.h"
#include <string>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio_wrapper.hpp"

extern "C" {

namespace pinocchio_wrapper {

struct State {
  pinocchio::Model *model = nullptr;
  pinocchio::Data *model_data = nullptr;
  pinocchio::FrameIndex ee_frame_idx;
  Eigen::VectorXd ik_sol_v;
  pinocchio::Data::Matrix6x ik_sol_J;
};

State *initialize(const char *ee_link_name, const char *xml_buffer) {
  auto model = new pinocchio::Model();
  pinocchio::urdf::buildModelFromXML(xml_buffer, *model);
  auto model_data = new pinocchio::Data(*model);

  return new State{model, model_data, model->getBodyId(ee_link_name),
                   Eigen::VectorXd(model->nv),
                   pinocchio::Data::Matrix6x(6, model->nv)};
}

void destroy(State *state) {
  delete state->model;
  delete state->model_data;
  delete state;
}

Eigen::VectorXd get_lower_position_limits(State *pinocchio_state) {
  return pinocchio_state->model->lowerPositionLimit;
}
Eigen::VectorXd get_upper_position_limits(State *pinocchio_state) {
  return pinocchio_state->model->upperPositionLimit;
}
Eigen::VectorXd get_velocity_limits(State *pinocchio_state) {
  return pinocchio_state->model->velocityLimit;
}
int get_nq(State *pinocchio_state) { return pinocchio_state->model->nq; }

Eigen::VectorXd forward_kinematics(State *pinocchio_state,
                                   const Eigen::VectorXd &q) {
  auto model = *pinocchio_state->model;
  auto model_data = *pinocchio_state->model_data;
  auto ee_frame_idx = pinocchio_state->ee_frame_idx;

  pinocchio::forwardKinematics(model, model_data, q);
  pinocchio::updateFramePlacement(model, model_data, ee_frame_idx);

  auto pos_data = model_data.oMf[ee_frame_idx].translation().transpose();
  auto quat_data = Eigen::Quaterniond(model_data.oMf[ee_frame_idx].rotation());

  Eigen::VectorXd result(7);
  for (int i = 0; i < 3; i++) {
    result[i] = pos_data[i];
  }
  result[3] = quat_data.x();
  result[4] = quat_data.y();
  result[5] = quat_data.z();
  result[6] = quat_data.w();

  return result;
}

void compute_jacobian(
    State *state, const Eigen::VectorXd &joint_positions,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>> &J) {
  auto model = *state->model;
  auto model_data = *state->model_data;
  auto ee_frame_idx = state->ee_frame_idx;
  pinocchio::computeFrameJacobian(model, model_data, joint_positions,
                                  ee_frame_idx, pinocchio::LOCAL_WORLD_ALIGNED,
                                  J);
}

Eigen::Matrix<double, Eigen::Dynamic, 1>
inverse_dynamics(State *state, const Eigen::VectorXd &q,
                 const Eigen::VectorXd &v, const Eigen::VectorXd &a) {
  auto model = *state->model;
  auto model_data = *state->model_data;
  auto ee_frame_idx = state->ee_frame_idx;
  return pinocchio::rnea(model, model_data, q, v, a);
}

void inverse_kinematics(State *state, const Eigen::Vector3d &ee_pos_,
                        const Eigen::Quaterniond &ee_quat_,
                        Eigen::VectorXd &ik_sol_p_, double eps,
                        int64_t max_iters, double dt, double damping) {
  auto model_ = *state->model;
  auto model_data_ = *state->model_data;
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
      spdlog::info("Ending IK at {}/{} iteration.", i + 1, max_iters);
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
    spdlog::warn("WARNING: IK did not converge!");
  }
}

} // namespace pinocchio_wrapper

} /* extern "C" */
