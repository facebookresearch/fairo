// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#define MAX_CIRCULAR_BUFFER_SIZE 300000 // 5 minutes of data at 1kHz
#define MAX_MODEL_BYTES 1048576         // 1 megabyte

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
struct RobotClientContext {
  long int last_update_ns = 0;
  RobotClientMetadata metadata;
  TorchScriptedController *default_controller = nullptr;
};

class ControllerManager {
public:
  ControllerManager(int num_dofs, std::vector<char> &default_controller_buffer);

  void controlUpdate(TorchRobotState torch_robot_state,
                     std::vector<float> desired_torque, std::string &error_msg);

  void setController(std::vector<char> &model_buffer, std::string &error_msg);

  void updateController(std::vector<char> &update_buffer,
                        std::string &error_msg);

  void terminateController(std::string &error_msg);

  void initRobotClient(RobotClientMetadata robot_client_metadata);
  RobotClientMetadata getRobotClientMetadata(void);

private:
  bool validRobotContext();

private:
  int num_dofs_;
  long int threshold_ns_ = THRESHOLD_NS;

  // Concurrency management
  ControllerStatus controller_status_;
  std::mutex service_mtx_;
  std::mutex controller_mutex_;

  // Logging
  CircularBuffer<TorchRobotState> robot_state_buffer_ =
      CircularBuffer<TorchRobotState>(MAX_CIRCULAR_BUFFER_SIZE);

  std::shared_ptr<TorchScriptedController> current_custom_controller_(nullptr);
};