// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "spdlog/spdlog.h"
#include <stdexcept>
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
  Eigen::VectorXd ik_sol_v;
  pinocchio::Data::Matrix6x ik_sol_J;
};

State *initialize(const char *xml_buffer) {
  auto model = new pinocchio::Model();
  pinocchio::urdf::buildModelFromXML(xml_buffer, *model);
  auto model_data = new pinocchio::Data(*model);

  return new State{model, model_data, Eigen::VectorXd(model->nv),
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
                                   const Eigen::VectorXd &q,
                                   int64_t frame_idx) {
  pinocchio::FrameIndex frame_idx_ =
      static_cast<pinocchio::FrameIndex>(frame_idx);
  auto model = *pinocchio_state->model;
  auto model_data = *pinocchio_state->model_data;

  pinocchio::forwardKinematics(model, model_data, q);
  pinocchio::updateFramePlacement(model, model_data, frame_idx_);

  auto pos_data = model_data.oMf[frame_idx_].translation().transpose();
  auto quat_data = Eigen::Quaterniond(model_data.oMf[frame_idx_].rotation());

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
                             Eigen::RowMajor>> &J,
    int64_t frame_idx) {
  pinocchio::FrameIndex frame_idx_ =
      static_cast<pinocchio::FrameIndex>(frame_idx);
  auto model = *state->model;
  auto model_data = *state->model_data;
  pinocchio::computeFrameJacobian(model, model_data, joint_positions,
                                  frame_idx_, pinocchio::LOCAL_WORLD_ALIGNED,
                                  J);
}

Eigen::Matrix<double, Eigen::Dynamic, 1>
inverse_dynamics(State *state, const Eigen::VectorXd &q,
                 const Eigen::VectorXd &v, const Eigen::VectorXd &a) {
  auto model = *state->model;
  auto model_data = *state->model_data;
  return pinocchio::rnea(model, model_data, q, v, a);
}

void inverse_kinematics(State *state, const Eigen::Vector3d &link_pos,
                        const Eigen::Quaterniond &link_quat, int64_t frame_idx,
                        Eigen::VectorXd &ik_sol_p_, double eps,
                        int64_t max_iters, double dt, double damping) {
  pinocchio::FrameIndex frame_idx_ =
      static_cast<pinocchio::FrameIndex>(frame_idx);
  auto model = *state->model;
  auto model_data = *state->model_data;
  auto ik_sol_v = state->ik_sol_v;
  auto ik_sol_J = state->ik_sol_J;

  auto link_orient = link_quat.toRotationMatrix();

  // Initialize IK variables
  const pinocchio::SE3 desired_ee(link_orient, link_pos);

  ik_sol_J.setZero();

  Eigen::Matrix<double, 6, 1> err;
  ik_sol_v.setZero();

  // Reset robot pose
  pinocchio::forwardKinematics(model, model_data, ik_sol_p_);
  pinocchio::updateFramePlacement(model, model_data, frame_idx_);

  // Solve IK iteratively
  for (int i = 0; i < max_iters; i++) {
    // Compute forward kinematics error
    pinocchio::forwardKinematics(model, model_data, ik_sol_p_);
    pinocchio::updateFramePlacement(model, model_data, frame_idx_);
    const pinocchio::SE3 dMf = desired_ee.actInv(model_data.oMf[frame_idx_]);
    err = pinocchio::log6(dMf).toVector();

    // Check termination
    if (err.norm() < eps) {
      break;
    }

    // Descent solution
    pinocchio::computeFrameJacobian(model, model_data, ik_sol_p_, frame_idx_,
                                    pinocchio::LOCAL, ik_sol_J);

    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = ik_sol_J * ik_sol_J.transpose();
    JJt.diagonal().array() += damping;
    ik_sol_v.noalias() = -ik_sol_J.transpose() * JJt.ldlt().solve(err);
    ik_sol_p_ = pinocchio::integrate(model, ik_sol_p_, ik_sol_v * dt);
  }

  if (err.norm() >= eps) {
    spdlog::warn("WARNING: IK did not converge!");
  }
}

int64_t get_link_idx_from_name(State *state, const char *link_name) {
  std::string link_name_(link_name);
  int64_t result = state->model->getBodyId(link_name_);
  if (result == state->model->nframes) {
    throw std::invalid_argument("Unknown link name " + link_name_);
  }
  return result;
}

char *get_link_name_from_idx(State *state, int64_t link_idx) {
  if (link_idx >= state->model->nframes) {
    throw std::invalid_argument("Invalid link index: " + link_idx);
  }
  return const_cast<char *>(state->model->frames[link_idx].name.c_str());
}

} // namespace pinocchio_wrapper

} /* extern "C" */
