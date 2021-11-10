// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <fstream>
#include <iostream>
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
  std::string ee_link_name_;

  RobotModelPinocchio(std::string urdf_filename, std::string ee_link_name) {
    ee_link_name_ = ee_link_name;

    std::ifstream stream(urdf_filename);
    xml_buffer_ = std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());

    initialize();
  }

  RobotModelPinocchio(std::vector<std::string> serialized_state) {
    ee_link_name_ = serialized_state[0];
    xml_buffer_ = serialized_state[1];
    initialize();
  }

  ~RobotModelPinocchio() { pinocchio_wrapper::destroy(pinocchio_state_); }

  void initialize() {
    pinocchio_state_ =
        pinocchio_wrapper::initialize(ee_link_name_, xml_buffer_);
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

  c10::List<torch::Tensor> forward_kinematics(torch::Tensor joint_positions) {
    c10::List<torch::Tensor> result;
    torch::Tensor pos_result = torch::zeros(3, torch::kFloat32);
    torch::Tensor quat_result = torch::zeros(4, torch::kFloat32);

    joint_positions = validTensor(joint_positions);
    std::cout << "1" << std::endl;
    auto result_intermediate = pinocchio_wrapper::forward_kinematics(
        pinocchio_state_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)));
    std::cout << "2" << std::endl;

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

  torch::Tensor compute_jacobian(torch::Tensor joint_positions) {
    std::cout << "a" << std::endl;
    int nq = pinocchio_wrapper::get_nq(pinocchio_state_);
    joint_positions = validTensor(joint_positions);
    std::cout << "b" << std::endl;

    torch::Tensor result = torch::zeros({6, nq}, torch::kFloat64);
    std::cout << "c" << std::endl;
    Eigen::Map<dtt::MatrixXrm<double>> J(result.data_ptr<double>(),
                                         result.size(0), result.size(1));
    std::cout << "d" << std::endl;
    pinocchio_wrapper::compute_jacobian(
        pinocchio_state_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)), J);
    std::cout << "e" << std::endl;

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

  torch::Tensor inverse_kinematics(torch::Tensor ee_pos, torch::Tensor ee_quat,
                                   torch::Tensor rest_pose, double eps = 1e-4,
                                   int64_t max_iters = 1000, double dt = 0.1,
                                   double damping = 1e-12) {
    ee_pos = validTensor(ee_pos);
    Eigen::Vector3d ee_pos_(
        Eigen::Map<Eigen::Vector3d>(ee_pos.data_ptr<double>(), 3));

    auto quat = tensor4ToQuat(ee_quat);
    auto ee_quat_ = Eigen::Quaterniond(quat).cast<double>();

    rest_pose = validTensor(rest_pose);
    ik_sol_p_ = matrixToVector(dtt::libtorch2eigen<double>(rest_pose));

    pinocchio_wrapper::inverse_kinematics(pinocchio_state_, ee_pos_, ee_quat_,
                                          ik_sol_p_, eps, max_iters, dt,
                                          damping);
    std::vector<int64_t> dims = {ik_sol_p_.rows()};
    return torch::from_blob(ik_sol_p_.data(), dims, torch::kFloat64).clone();
  }
};

TORCH_LIBRARY(torchscript_pinocchio, m) {
  m.class_<RobotModelPinocchio>("RobotModelPinocchio")
      .def(torch::init<std::string, std::string>())
      .def("get_joint_angle_limits",
           &RobotModelPinocchio::get_joint_angle_limits)
      .def("get_joint_velocity_limits",
           &RobotModelPinocchio::get_joint_velocity_limits)
      .def("forward_kinematics", &RobotModelPinocchio::forward_kinematics)
      .def("compute_jacobian", &RobotModelPinocchio::compute_jacobian)
      .def("inverse_dynamics", &RobotModelPinocchio::inverse_dynamics)
      .def("inverse_kinematics", &RobotModelPinocchio::inverse_kinematics)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RobotModelPinocchio> &self)
              -> std::vector<std::string> {
            return std::vector<std::string>{self->ee_link_name_,
                                            self->xml_buffer_};
          },
          // __setstate__
          [](std::vector<std::string> state)
              -> c10::intrusive_ptr<RobotModelPinocchio> {
            return c10::make_intrusive<RobotModelPinocchio>(std::move(state));
          });
}

} /* extern "C" */
