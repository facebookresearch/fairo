// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "torch_server_ops.hpp"
#include <torch/script.h>
#include <vector>

extern "C" {

TorchRobotState::TorchRobotState(int num_dofs) {
  num_dofs_ = num_dofs;
  // Create initial state dictionary
  rs_timestamp_ = torch::zeros(2).to(torch::kInt32);
  rs_joint_positions_ = torch::zeros(num_dofs);
  rs_joint_velocities_ = torch::zeros(num_dofs);
  rs_motor_torques_measured_ = torch::zeros(num_dofs);
  rs_motor_torques_external_ = torch::zeros(num_dofs);

  state_dict_.insert("timestamp", rs_timestamp_);
  state_dict_.insert("joint_positions", rs_joint_positions_);
  state_dict_.insert("joint_velocities", rs_joint_velocities_);
  state_dict_.insert("motor_torques_measured", rs_motor_torques_measured_);
  state_dict_.insert("motor_torques_external", rs_motor_torques_external_);

  input_.push_back(state_dict_);
}

void TorchRobotState::update_state(int timestamp_s, int timestamp_ns,
                                   std::vector<float> joint_positions,
                                   std::vector<float> joint_velocities,
                                   std::vector<float> motor_torques_measured,
                                   std::vector<float> motor_torques_external) {
  rs_timestamp_[0] = timestamp_s;
  rs_timestamp_[1] = timestamp_ns;
  for (int i = 0; i < joint_positions.size(); i++) {
    rs_joint_positions_[i] = joint_positions[i];
    rs_joint_velocities_[i] = joint_velocities[i];
    rs_motor_torques_measured_[i] = motor_torques_measured[i];
    rs_motor_torques_external_[i] = motor_torques_external[i];
  }
}

TorchScriptedController::TorchScriptedController(std::istream &stream) {
  module_ = torch::jit::load(stream);
}

std::vector<float> TorchScriptedController::forward(TorchRobotState &input) {
  // Step controller & generate torque command response
  c10::Dict<torch::jit::IValue, torch::jit::IValue> controller_state_dict =
      module_.forward(input.input_).toGenericDict();

  torch::jit::IValue key = torch::jit::IValue("joint_torques");
  torch::Tensor desired_torque = controller_state_dict.at(key).toTensor();

  std::vector<float> result;
  for (int i = 0; i < input.num_dofs_; i++) {
    result.push_back(desired_torque[i].item<float>());
  }
  return result;
}

bool TorchScriptedController::is_terminated() {
  return module_.get_method("is_terminated")(empty_input_).toBool();
}

void TorchScriptedController::reset() {
  module_.get_method("reset")(empty_input_);
}

bool TorchScriptedController::param_dict_load(std::istream &model_stream) {
  torch::jit::script::Module param_dict_container;
  try {
    param_dict_container = torch::jit::load(model_stream);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the param container:\n";
    std::cerr << e.msg() << std::endl;
    return false;
  }

  // Create controller update input dict
  param_dict_input_.clear();
  param_dict_input_.push_back(param_dict_container.forward(empty_input_));

  return true;
}

void TorchScriptedController::param_dict_update_module() {
  module_.get_method("update")(param_dict_input_);
}

} /* extern "C" */
