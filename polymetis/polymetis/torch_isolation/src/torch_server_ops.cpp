// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "torch_server_ops.hpp"
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

extern "C" {

struct TorchTensor {
  torch::Tensor data;
};
struct TorchScriptModule {
  torch::jit::script::Module data;
};
struct TorchInput {
  std::vector<torch::jit::IValue> data;
};

struct StateDict {
  c10::Dict<std::string, torch::Tensor> data;
};

TorchRobotState::TorchRobotState(int num_dofs) {
  num_dofs_ = num_dofs;

  // Create initial state dictionary
  rs_timestamp_ = new TorchTensor{torch::zeros(2).to(torch::kInt32)};
  rs_joint_positions_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_joint_velocities_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_measured_ = new TorchTensor{torch::zeros(num_dofs)};
  rs_motor_torques_external_ = new TorchTensor{torch::zeros(num_dofs)};

  state_dict_ = new StateDict{c10::Dict<std::string, torch::Tensor>()};

  state_dict_->data.insert("timestamp", rs_timestamp_->data);
  state_dict_->data.insert("joint_positions", rs_joint_positions_->data);
  state_dict_->data.insert("joint_velocities", rs_joint_velocities_->data);
  state_dict_->data.insert("motor_torques_measured",
                           rs_motor_torques_measured_->data);
  state_dict_->data.insert("motor_torques_external",
                           rs_motor_torques_external_->data);

  input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  input_->data.push_back(state_dict_->data);
}

TorchRobotState::~TorchRobotState() {
  delete rs_timestamp_;
  delete rs_joint_positions_;
  delete rs_joint_velocities_;
  delete rs_motor_torques_measured_;
  delete rs_motor_torques_external_;
  delete state_dict_;
  delete input_;
}

void TorchRobotState::update_state(int timestamp_s, int timestamp_ns,
                                   std::vector<float> joint_positions,
                                   std::vector<float> joint_velocities,
                                   std::vector<float> motor_torques_measured,
                                   std::vector<float> motor_torques_external) {
  rs_timestamp_->data[0] = timestamp_s;
  rs_timestamp_->data[1] = timestamp_ns;
  for (int i = 0; i < joint_positions.size(); i++) {
    rs_joint_positions_->data[i] = joint_positions[i];
    rs_joint_velocities_->data[i] = joint_velocities[i];
    rs_motor_torques_measured_->data[i] = motor_torques_measured[i];
    rs_motor_torques_external_->data[i] = motor_torques_external[i];
  }
}

TorchScriptedController::TorchScriptedController(std::istream &stream) {
  module_ = new TorchScriptModule{torch::jit::load(stream)};

  param_dict_input_ = new TorchInput{std::vector<torch::jit::IValue>()};
  empty_input_ = new TorchInput{std::vector<torch::jit::IValue>()};
}

TorchScriptedController::~TorchScriptedController() {
  delete module_;
  delete param_dict_input_;
  delete empty_input_;
}

std::vector<float> TorchScriptedController::forward(TorchRobotState &input) {
  // Step controller & generate torque command response
  c10::Dict<torch::jit::IValue, torch::jit::IValue> controller_state_dict =
      module_->data.forward(input.input_->data).toGenericDict();

  torch::jit::IValue key = torch::jit::IValue("joint_torques");
  torch::Tensor desired_torque = controller_state_dict.at(key).toTensor();

  std::vector<float> result;
  for (int i = 0; i < input.num_dofs_; i++) {
    result.push_back(desired_torque[i].item<float>());
  }
  return result;
}

bool TorchScriptedController::is_terminated() {
  return module_->data.get_method("is_terminated")(empty_input_->data).toBool();
}

void TorchScriptedController::reset() {
  module_->data.get_method("reset")(empty_input_->data);
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
  param_dict_input_->data.clear();
  param_dict_input_->data.push_back(
      param_dict_container.forward(empty_input_->data));

  return true;
}

void TorchScriptedController::param_dict_update_module() {
  module_->data.get_method("update")(param_dict_input_->data);
}

} /* extern "C" */
