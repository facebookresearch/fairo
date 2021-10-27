// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define C_TORCH_EXPORT __attribute__((visibility("default")))

#ifndef TORCH_SERVER_OPS_H
#define TORCH_SERVER_OPS_H

struct TorchTensor;       // torch::Tensor
struct TorchScriptModule; // torch::jit::script::Module
struct TorchInput;        // std::vector<torch::jit::IValue>
struct StateDict;         // c10::Dict<std::string, struct TorchTensor>

class TorchRobotState {
private:
  struct StateDict *state_dict_;

  // Robot states
  struct TorchTensor *rs_timestamp_;
  struct TorchTensor *rs_joint_positions_;
  struct TorchTensor *rs_joint_velocities_;
  struct TorchTensor *rs_motor_torques_measured_;
  struct TorchTensor *rs_motor_torques_external_;

public:
  TorchRobotState(int num_dofs);
  void update_state(int timestamp_s, int timestamp_ns,
                    std::vector<float> joint_positions,
                    std::vector<float> joint_velocities,
                    std::vector<float> motor_torques_measured,
                    std::vector<float> motor_torques_external);
  struct TorchInput *input_;
  int num_dofs_;
};

class TorchScriptedController {
private:
  struct TorchScriptModule *module_;
  TorchRobotState robot_state_ = TorchRobotState(1);

  // Inputs
  struct TorchInput *param_dict_input_;
  struct TorchInput *empty_input_;

public:
  TorchScriptedController(std::istream &stream);

  std::vector<float> forward(TorchRobotState &input);

  bool param_dict_load(std::istream &stream);
  void param_dict_update_module();

  bool is_terminated();
  void reset();
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
