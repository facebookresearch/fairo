// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef polymetis_SERVER_H
#define polymetis_SERVER_H

#include "spdlog/spdlog.h"
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "polymetis.grpc.pb.h"

#include "polymetis/utils.h"
#include "torch_server_ops.hpp"
#include "yaml-cpp/yaml.h"

#define MAX_CIRCULAR_BUFFER_SIZE 300000 // 5 minutes of data at 1kHz
#define MAX_MODEL_BYTES 1048576         // 1 megabyte
#define THRESHOLD_NS 1000000000         // 1s
#define SPIN_INTERVAL_USEC 20000        // 0.02s (50hz)
#define RT_LOW_PRIO 40

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using grpc::StatusCode;

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
  std::unique_ptr<TorchScriptedController> custom_controller;
};

/**
TODO
*/
struct RobotClientContext {
  long int last_update_ns = 0;
  RobotClientMetadata metadata;
  TorchScriptedController *default_controller = nullptr;
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

  int setThreadPriority(int prio);

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

  std::unique_ptr<TorchRobotState> torch_robot_state_;
};

#endif