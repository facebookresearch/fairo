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

#include "polymetis/controller_manager.hpp"

#define MAX_MODEL_BYTES 1048576  // 1 megabyte
#define SPIN_INTERVAL_USEC 20000 // 0.02s (50hz)

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
class PolymetisControllerServerImpl final
    : public PolymetisControllerServer::Service {
public:
  /**
  TODO
  */
  explicit PolymetisControllerServerImpl();

  bool validRobotContext(void);

  // Robot client methods

  /**
  TODO
  */
  void initRobotClient(const RobotClientMetadata *robot_client_metadata);

  /**
  TODO
  */
  void controlUpdate(const RobotState *robot_state,
                     TorqueCommand *torque_command);

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

  std::mutex service_mtx_;

  ControllerManager controller_manager_;
};

#endif