// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <fstream>
#include <iostream>
#include <string>

#include "dtt.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <torch/script.h>
#include <torch/torch.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"

// IK parameters
#define IK_EPS 1e-4
#define IK_IT_MAX 1000
#define IK_DT 1e-1
#define IK_DAMP 1e-6

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
  pinocchio::Model model_;
  pinocchio::Data model_data_;
  pinocchio::FrameIndex ee_idx_;

  std::string xml_buffer_;
  std::string ee_joint_name_;

  RobotModelPinocchio(std::string urdf_filename, std::string ee_joint_name) {
    ee_joint_name_ = ee_joint_name;

    std::ifstream stream(urdf_filename);
    xml_buffer_ = std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());

    initialize();
  }

  RobotModelPinocchio(std::vector<std::string> serialized_state) {
    ee_joint_name_ = serialized_state[0];
    xml_buffer_ = serialized_state[1];
    initialize();
  }

  void initialize() {
    pinocchio::urdf::buildModelFromXML(xml_buffer_, model_);
    model_data_ = pinocchio::Data(model_);
    ee_idx_ = model_.getFrameId(ee_joint_name_);
  }

  c10::List<torch::Tensor> get_joint_angle_limits(void) {
    c10::List<torch::Tensor> result;
    torch::Tensor l_result = torch::zeros(model_.nq, torch::kFloat32);
    torch::Tensor u_result = torch::zeros(model_.nq, torch::kFloat32);

    for (int i = 0; i < model_.nq; i++) {
      l_result[i] = model_.lowerPositionLimit[i];
      u_result[i] = model_.upperPositionLimit[i];
    }
    result.push_back(l_result);
    result.push_back(u_result);

    return result;
  }

  torch::Tensor get_joint_velocity_limits(void) {
    torch::Tensor result = torch::zeros(model_.nq, torch::kFloat32);

    for (int i = 0; i < model_.nq; i++) {
      result[i] = model_.velocityLimit[i];
    }

    return result;
  }

  c10::List<torch::Tensor> forward_kinematics(torch::Tensor joint_positions) {
    c10::List<torch::Tensor> result;
    torch::Tensor pos_result = torch::zeros(3, torch::kFloat32);
    torch::Tensor quat_result = torch::zeros(4, torch::kFloat32);

    joint_positions = validTensor(joint_positions);
    pinocchio::forwardKinematics(
        model_, model_data_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)));
    pinocchio::updateFramePlacement(model_, model_data_, ee_idx_);

    auto pos_data = model_data_.oMf[ee_idx_].translation().transpose();
    auto quat_data = Eigen::Quaterniond(model_data_.oMf[ee_idx_].rotation());

    for (int i = 0; i < 3; i++) {
      pos_result[i] = pos_data[i];
    }
    quat_result[0] = quat_data.x();
    quat_result[1] = quat_data.y();
    quat_result[2] = quat_data.z();
    quat_result[3] = quat_data.w();

    result.push_back(pos_result);
    result.push_back(quat_result);

    return result;
  }

  torch::Tensor compute_jacobian(torch::Tensor joint_positions) {
    joint_positions = validTensor(joint_positions);

    torch::Tensor result = torch::zeros({6, model_.nq}, torch::kFloat64);
    Eigen::Map<dtt::MatrixXrm<double>> J(result.data_ptr<double>(),
                                         result.size(0), result.size(1));
    pinocchio::computeFrameJacobian(
        model_, model_data_,
        matrixToVector(dtt::libtorch2eigen<double>(joint_positions)), ee_idx_,
        pinocchio::LOCAL_WORLD_ALIGNED, J);

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
        pinocchio::rnea(model_, model_data_, q, v, a);
    std::vector<int64_t> dims = {tau.rows()};
    return torch::from_blob(tau.data(), dims, torch::kFloat64).clone();
  }

  torch::Tensor inverse_kinematics(torch::Tensor ee_pos,
                                   torch::Tensor ee_quat) {
    const pinocchio::SE3 oMdes(Eigen::Matrix3d::Identity(),
                               Eigen::Vector3d(1., 0., 1.));
    torch::Tensor result = torch::zeros(model_.nq, torch::kFloat32);

    // Initialize IK variables
    Eigen::VectorXd q = pinocchio::neutral(model_);

    pinocchio::Data::Matrix6x J(6, model_.nv);
    J.setZero();

    Eigen::Matrix<double, 6, 1> err;
    Eigen::VectorXd v(model_.nv);

    // Solve IK iteratively
    for (int i = 0; i < IK_IT_MAX; i++) {
      // Compute forward kinematics error
      pinocchio::forwardKinematics(model_, model_data_, q);
      const pinocchio::SE3 dMi = oMdes.actInv(model_data_.oMi[ee_idx_]);
      err = pinocchio::log6(dMi).toVector();

      // Check termination
      if (err.norm() < IK_EPS) {
        break;
      }

      // Descent solution
      pinocchio::computeJointJacobian(model_, model_data_, q, ee_idx_, J);
      pinocchio::Data::Matrix6 JJt;
      JJt.noalias() = J * J.transpose();
      JJt.diagonal().array() += IK_DAMP;
      v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
      q = pinocchio::integrate(model_, q, v * IK_DT);
    }

    for (int i = 0; i < model_.nq; i++) {
      result[i] = q[i];
    }

    return result;
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
            return std::vector<std::string>{self->ee_joint_name_,
                                            self->xml_buffer_};
          },
          // __setstate__
          [](std::vector<std::string> state)
              -> c10::intrusive_ptr<RobotModelPinocchio> {
            return c10::make_intrusive<RobotModelPinocchio>(std::move(state));
          });
}