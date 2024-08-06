// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <torch/script.h>
#include <torch/torch.h>

#include "dtt.h"
#include "pinocchio_wrapper.hpp"
#include "rotations.hpp"

extern "C" {

torch::Tensor validTensor(torch::Tensor x) {
  if (x.dim() < 2) {
    x = x.unsqueeze(1);
  }
  return x.to(torch::kDouble);
}

Eigen::VectorXd matrixToVector(Eigen::MatrixXd A) {
  return Eigen::VectorXd(
      Eigen::Map<Eigen::VectorXd>(A.data(), A.cols() * A.rows()));
}

struct RobotModelPinocchio : torch::CustomClassHolder {
  Eigen::VectorXd ik_sol_p_;
  Eigen::VectorXd ik_sol_v_;

  pinocchio_wrapper::State *pinocchio_state_ = nullptr;

  std::string xml_buffer_;

  RobotModelPinocchio(std::string urdf_filename_or_serialized_state,
                      bool serialized_state = false) {
    if (serialized_state) {
      xml_buffer_ = urdf_filename_or_serialized_state;
    } else {
      std::ifstream stream(urdf_filename_or_serialized_state);
      std::stringstream buffer;
      buffer << stream.rdbuf();
      xml_buffer_ = buffer.str();
    }

    initialize();
  }

  ~RobotModelPinocchio() { pinocchio_wrapper::destroy(pinocchio_state_); }

  void initialize() {
    pinocchio_state_ = pinocchio_wrapper::initialize(xml_buffer_.c_str());
  }

  c10::List<torch::Tensor> get_joint_angle_limits(void) {
    c10::List<torch::Tensor> result;
    int nq = pinocchio_wrapper::get_nq(pinocchio_state_);
    torch::Tensor l_result = torch::zeros(nq, torch::kFloat32);
    torch::Tensor u_result = torch::zeros(nq, torch::kFloat32);

    auto lower_limit =
        pinocchio_wrapper::get_lower_position_limits(pinocchio_state_);
    auto upper_limit =
        pinocchio_wrapper::get_upper_position_limits(pinocchio_state_);

    for (int i = 0; i < nq; i++) {
      l_result[i] = lower_limit[i];
      u_result[i] = upper_limit[i];
    }
    result.push_back(l_result);
    result.push_back(u_result);

    return result;
  }

  torch::Tensor get_joint_velocity_limits(void) {
    int nq = pinocchio_wrapper::get_nq(pinocchio_state_);
    torch::Tensor result = torch::zeros(nq, torch::kFloat32);
    auto velocity_limit =
        pinocchio_wrapper::get_velocity_limits(pinocchio_state_);

    for (int i = 0; i < nq; i++) {
      result[i] = velocity_limit[i];
    }

    return result;
  }

  c10::List<torch::Tensor> forward_kinematics(torch::Tensor joint_positions,
                                              int64_t frame_idx) {
    c10::List<torch::Tensor> result;
    torch::Tensor pos_result = torch::zeros(3, torch::kFloat32);
    torch::Tensor quat_result = torch::zeros(4, torch::kFloat32);

    joint_positions = validTensor(joint_positions);
    auto result_intermediate = pinocchio_wrapper::forward_kinematics(
        pinocchio_state_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)),
        frame_idx);

    for (int i = 0; i < 3; i++) {
      pos_result[i] = result_intermediate[i];
    }
    for (int i = 0; i < 4; i++) {
      quat_result[i] = result_intermediate[i + 3];
    }

    result.push_back(pos_result);
    result.push_back(quat_result);

    return result;
  }

  torch::Tensor compute_jacobian(torch::Tensor joint_positions,
                                 int64_t frame_idx) {
    int nq = pinocchio_wrapper::get_nq(pinocchio_state_);
    joint_positions = validTensor(joint_positions);

    torch::Tensor result = torch::zeros({6, nq}, torch::kFloat64);
    Eigen::Map<dtt::MatrixXrm<double>> J(result.data_ptr<double>(),
                                         result.size(0), result.size(1));
    pinocchio_wrapper::compute_jacobian(
        pinocchio_state_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)), J,
        frame_idx);

    return result;
  }

  torch::Tensor inverse_dynamics(torch::Tensor joint_positions,
                                 torch::Tensor joint_velocities,
                                 torch::Tensor joint_accelerations) {
    joint_positions = validTensor(joint_positions);
    joint_velocities = validTensor(joint_velocities);
    joint_accelerations = validTensor(joint_accelerations);
    auto q = matrixToVector(dtt::libtorch2eigen<double>(joint_positions));
    auto v = matrixToVector(dtt::libtorch2eigen<double>(joint_velocities));
    auto a = matrixToVector(dtt::libtorch2eigen<double>(joint_accelerations));

    Eigen::Matrix<double, Eigen::Dynamic, 1> tau =
        pinocchio_wrapper::inverse_dynamics(pinocchio_state_, q, v, a);
    std::vector<int64_t> dims = {tau.rows()};
    return torch::from_blob(tau.data(), dims, torch::kFloat64).clone();
  }

  torch::Tensor inverse_kinematics(torch::Tensor link_pos,
                                   torch::Tensor link_quat, int64_t frame_idx,
                                   torch::Tensor rest_pose, double eps = 1e-4,
                                   int64_t max_iters = 1000, double dt = 0.1,
                                   double damping = 1e-12) {
    link_pos = validTensor(link_pos);
    Eigen::Vector3d link_pos_(
        Eigen::Map<Eigen::Vector3d>(link_pos.data_ptr<double>(), 3));

    auto quat = tensor4ToQuat(link_quat);
    auto link_quat_ = Eigen::Quaterniond(quat).cast<double>();

    rest_pose = validTensor(rest_pose);
    ik_sol_p_ = matrixToVector(dtt::libtorch2eigen<double>(rest_pose));

    pinocchio_wrapper::inverse_kinematics(pinocchio_state_, link_pos_,
                                          link_quat_, frame_idx, ik_sol_p_, eps,
                                          max_iters, dt, damping);
    std::vector<int64_t> dims = {ik_sol_p_.rows()};
    return torch::from_blob(ik_sol_p_.data(), dims, torch::kFloat64).clone();
  }

  int64_t get_link_idx_from_name(std::string link_name) {
    return pinocchio_wrapper::get_link_idx_from_name(pinocchio_state_,
                                                     link_name.c_str());
  }

  std::string get_link_name_from_idx(int64_t link_idx) {
    std::string result(
        pinocchio_wrapper::get_link_name_from_idx(pinocchio_state_, link_idx));
    return result;
  }
};

TORCH_LIBRARY(torchscript_pinocchio, m) {
  m.class_<RobotModelPinocchio>("RobotModelPinocchio")
      .def(torch::init<std::string, bool>())
      .def("get_joint_angle_limits",
           &RobotModelPinocchio::get_joint_angle_limits)
      .def("get_joint_velocity_limits",
           &RobotModelPinocchio::get_joint_velocity_limits)
      .def("forward_kinematics", &RobotModelPinocchio::forward_kinematics)
      .def("compute_jacobian", &RobotModelPinocchio::compute_jacobian)
      .def("inverse_dynamics", &RobotModelPinocchio::inverse_dynamics)
      .def("inverse_kinematics", &RobotModelPinocchio::inverse_kinematics)
      .def("get_link_idx_from_name",
           &RobotModelPinocchio::get_link_idx_from_name)
      .def("get_link_name_from_idx",
           &RobotModelPinocchio::get_link_name_from_idx)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RobotModelPinocchio> &self)
              -> std::string { return std::string{self->xml_buffer_}; },
          // __setstate__
          [](std::string state) -> c10::intrusive_ptr<RobotModelPinocchio> {
            return c10::make_intrusive<RobotModelPinocchio>(std::move(state),
                                                            true);
          });
}

} /* extern "C" */
