// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef polymetis_SERVER_H
#define polymetis_SERVER_H

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include <sys/ipc.h>
#include <sys/shm.h>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "polymetis.grpc.pb.h"

#include "polymetis/utils.h"
#include "yaml-cpp/yaml.h"

#include <torch/script.h>

#define MAX_CIRCULAR_BUFFER_SIZE 300000 // 5 minutes of data at 1kHz
#define MAX_MODEL_BYTES 1048576         // 1 megabyte
#define THRESHOLD_NS 1000000000         // 1s
#define SPIN_INTERVAL_USEC 20000        // 0.02s (50hz)
#define SHM_SIZE 65536                  // 64K

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

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
  ServerContext *server_context;
  uint episode_begin = -1;
  uint episode_end = -1;
  uint timestep = 0;
  ControllerStatus status = UNINITIALIZED;
  std::mutex controller_mtx;
  torch::jit::script::Module custom_controller;
};

/**
TODO
*/
struct RobotClientContext {
  long int last_update_ns = 0;
  RobotClientMetadata metadata;
  torch::jit::script::Module default_controller;
};

/**
TODO
*/
class PolymetisControllerServerImpl final
    : public PolymetisControllerServer::Service {
public:
  /**
  TODO
  */
  explicit PolymetisControllerServerImpl();

  /**
  TODO
  */
  bool validRobotContext();

  void resetControllerContext();

  // Robot client methods

  /**
  TODO
  */
  Status InitRobotClient(ServerContext *context,
                         const RobotClientMetadata *robot_client_metadata,
                         Empty *) override;

  /**
  TODO
  */
  Status ControlUpdate(ServerContext *context, const RobotState *robot_state,
                       TorqueCommand *torque_command) override;

  // User client methods

  /**
  TODO
  */
  Status GetRobotClientMetadata(ServerContext *context, const Empty *,
                                RobotClientMetadata *metadata) override;

  /**
  TODO
  */
  Status GetRobotState(ServerContext *context, const Empty *,
                       RobotState *robot_state) override;

  /**
  TODO
  */
  Status GetRobotStateStream(ServerContext *context, const Empty *,
                             ServerWriter<RobotState> *writer) override;

  /**
  TODO
  */
  Status GetRobotStateLog(ServerContext *context, const LogInterval *interval,
                          ServerWriter<RobotState> *writer) override;

  /**
  TODO
  */
  Status SetController(ServerContext *context,
                       ServerReader<ControllerChunk> *stream,
                       LogInterval *interval) override;

  /**
  TODO
  */
  Status UpdateController(ServerContext *context,
                          ServerReader<ControllerChunk> *stream,
                          LogInterval *interval) override;

  /**
  TODO
  */
  Status TerminateController(ServerContext *context, const Empty *,
                             LogInterval *interval) override;

  /**
  TODO
  */
  Status GetEpisodeInterval(ServerContext *context, const Empty *,
                            LogInterval *interval) override;

private:
  std::vector<char> controller_model_buffer_; // buffer for loading controllers
  std::vector<char>
      updates_model_buffer_; // buffer for loading controller update params
  int num_dofs_;
  long int threshold_ns_ = THRESHOLD_NS;

  std::mutex service_mtx_;

  CircularBuffer<RobotState> robot_state_buffer_ =
      CircularBuffer<RobotState>(MAX_CIRCULAR_BUFFER_SIZE);

  CustomControllerContext custom_controller_context_;
  RobotClientContext robot_client_context_;
  boost::interprocess::managed_shared_memory segment_;

  // Robot states
  torch::Tensor rs_timestamp_;
  torch::Tensor rs_joint_positions_;
  torch::Tensor rs_joint_velocities_;
  torch::Tensor rs_motor_torques_measured_;
  torch::Tensor rs_motor_torques_external_;

  c10::Dict<std::string, torch::Tensor> state_dict_;

  // Inputs
  std::vector<torch::jit::IValue> input_;
  std::vector<torch::jit::IValue> empty_input_;
  std::vector<torch::jit::IValue> param_dict_input_;
};

#endif