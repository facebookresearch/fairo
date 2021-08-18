// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef TORCH_SERVER_OPS_H
#define TORCH_SERVER_OPS_H

#include <torch/script.h>

class TorchRobotState {
private:
  c10::Dict<std::string, torch::Tensor> state_dict_;

  // Robot states
  torch::Tensor rs_timestamp_;
  torch::Tensor rs_joint_positions_;
  torch::Tensor rs_joint_velocities_;
  torch::Tensor rs_motor_torques_measured_;
  torch::Tensor rs_motor_torques_external_;

public:
  TorchRobotState(int num_dofs);
  void update_state(int timestamp_s, int timestamp_ns,
                    std::vector<float> joint_positions,
                    std::vector<float> joint_velocities,
                    std::vector<float> motor_torques_measured,
                    std::vector<float> motor_torques_external);

  int num_dofs_;
  std::vector<torch::jit::IValue> input_;
};

class TorchScriptedController {
private:
  torch::jit::script::Module module_;
  TorchRobotState robot_state_ = TorchRobotState(1);

  // Inputs
  std::vector<torch::jit::IValue> param_dict_input_;
  std::vector<torch::jit::IValue> empty_input_;

public:
  TorchScriptedController(std::istream &stream);

  std::vector<float> forward(TorchRobotState &input);

  bool param_dict_load(std::istream &stream);
  void param_dict_update_module();

  bool is_terminated();
  void reset();
};

#endif
