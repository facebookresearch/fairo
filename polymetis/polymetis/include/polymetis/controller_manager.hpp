// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef CONTROLLER_MANAGER_H
#define CONTROLLER_MANAGER_H

#include "spdlog/spdlog.h"
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "polymetis.grpc.pb.h"

#include "polymetis/utils.h"
#include "torch_server_ops.hpp"
#include "yaml-cpp/yaml.h"

#define MAX_CIRCULAR_BUFFER_SIZE 300000 // 5 minutes of data at 1kHz
#define THRESHOLD_NS 1000000000         // 1s
#define SPIN_INTERVAL_USEC 20000        // 0.02s (50hz)

/**
TODO
*/
enum ControllerStatus {
  UNINITIALIZED,
  READY,
  RUNNING,
  TERMINATING,
  TERMINATED
};

/**
TODO
*/
struct CustomControllerContext {
  uint episode_begin = -1;
  uint episode_end = -1;
  uint timestep = 0;
  ControllerStatus status = UNINITIALIZED;
  std::mutex controller_mtx;
  TorchScriptedController *custom_controller = nullptr;

  ~CustomControllerContext() { delete custom_controller; }
};

/**
TODO
*/
struct RobotClientContext {
  long int last_update_ns = 0;
  RobotClientMetadata metadata;
  TorchScriptedController *default_controller = nullptr;
};

class ControllerManager {
public:
  // Initialization methods

  ControllerManager(){};

  void initRobotClient(const RobotClientMetadata *robot_client_metadata,
                       std::string &error_msg);

  RobotClientMetadata getRobotClientMetadata(RobotClientMetadata *metadata,
                                             std::string &error_msg);

  // Log querying methods

  RobotState *getStateByBufferIndex(int index);

  int getStateBufferSize(void);

  void getEpisodeInterval(LogInterval *interval);

  // Interface methods

  void controlUpdate(const RobotState *robot_state,
                     std::vector<float> &desired_torque,
                     std::string &error_msg);

  void setController(std::vector<char> &model_buffer, LogInterval *interval,
                     std::string &error_msg);

  void updateController(std::vector<char> &update_buffer, LogInterval *interval,
                        std::string &error_msg);

  void terminateController(LogInterval *interval, std::string &error_msg);

private:
  // Helper methods

  void resetControllerContext(void);

  bool validRobotContext(void);

private:
  int num_dofs_;
  long int threshold_ns_ = THRESHOLD_NS;

  std::unique_ptr<TorchRobotState> torch_robot_state_;
  CircularBuffer<RobotState> robot_state_buffer_ =
      CircularBuffer<RobotState>(MAX_CIRCULAR_BUFFER_SIZE);

  CustomControllerContext custom_controller_context_;
  RobotClientContext robot_client_context_;
};

#endif