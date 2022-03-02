// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef TORCH_SERVER_OPS_H
#define TORCH_SERVER_OPS_H

#include <map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define C_TORCH_EXPORT __attribute__((visibility("default")))
#define WARM_UP_ITERS 5

struct TorchTensor;       // torch::Tensor
struct TorchScriptModule; // torch::jit::script::Module
struct TorchInput;        // std::vector<torch::jit::IValue>
struct StateDict;         // c10::Dict<std::string, struct TorchTensor>

class C_TORCH_EXPORT TorchRobotState {
private:
  struct StateDict *state_dict_ = nullptr;

  // Robot states
  struct TorchTensor *rs_timestamp_ = nullptr;
  struct TorchTensor *rs_joint_positions_ = nullptr;
  struct TorchTensor *rs_joint_velocities_ = nullptr;
  struct TorchTensor *rs_motor_torques_measured_ = nullptr;
  struct TorchTensor *rs_motor_torques_external_ = nullptr;

public:
  TorchRobotState(int num_dofs);
  ~TorchRobotState();
  void update_state(int timestamp_s, int timestamp_ns,
                    std::vector<float> joint_positions,
                    std::vector<float> joint_velocities,
                    std::vector<float> motor_torques_measured,
                    std::vector<float> motor_torques_external);
  struct TorchInput *input_ = nullptr;
  int num_dofs_;
};

class C_TORCH_EXPORT TorchScriptedController {
private:
  struct TorchScriptModule *module_ = nullptr;
  TorchRobotState robot_state_ = TorchRobotState(1);

  // Inputs
  struct TorchInput *param_dict_input_ = nullptr;
  struct TorchInput *empty_input_ = nullptr;

public:
  TorchScriptedController(char *data, size_t size,
                          TorchRobotState &init_robot_state);
  ~TorchScriptedController();

  void warmup_controller(int warmup_iters, TorchRobotState &init_robot_state);

  std::vector<float> forward(TorchRobotState &input);

  bool param_dict_load(char *data, size_t size);
  void param_dict_update_module();

  bool is_terminated();
  void reset();
};

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif // TORCH_SERVER_OPS_H
