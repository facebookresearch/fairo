// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#define MAX_CIRCULAR_BUFFER_SIZE 300000 // 5 minutes of data at 1kHz
#define MAX_MODEL_BYTES 1048576         // 1 megabyte
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

  ControllerManager(int num_dofs, std::vector<char> &default_controller_buffer);

  void initRobotClient(RobotClientMetadata robot_client_metadata);

  RobotClientMetadata getRobotClientMetadata(void);

  // Interface methods

  void controlUpdate(const RobotState *robot_state,
                     std::vector<float> &desired_torque,
                     std::string &error_msg);

  void setController(std::vector<char> &model_buffer, std::string &error_msg);

  void updateController(std::vector<char> &update_buffer,
                        std::string &error_msg);

  void terminateController(std::string &error_msg);

  // Log querying methods

  void getRobotState(); // TODO

  void awaitRobotState(); // TODO

private:
  bool validRobotContext();

private:
  int num_dofs_;
  long int threshold_ns_ = THRESHOLD_NS;

  std::mutex service_mtx_;

  CircularBuffer<RobotState> robot_state_buffer_ =
      CircularBuffer<RobotState>(MAX_CIRCULAR_BUFFER_SIZE);

  CustomControllerContext custom_controller_context_;
  RobotClientContext robot_client_context_;
};