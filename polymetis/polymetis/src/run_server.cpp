// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "polymetis/clients/franka_panda_client.hpp"

#include "yaml-cpp/yaml.h"

#include "polymetis/polymetis_server.hpp"
#include "real_time.hpp"
#include "torch_server_ops.hpp"

struct ServerInfo {
  YAML::Node cfg;
  std::shared_ptr<PolymetisControllerServerImpl> server_ptr;
};

void *RunDriver(void *server_info_ptr) {
  ServerInfo &server_info = *(static_cast<ServerInfo *>(server_info_ptr));

  // Launch franka client
  std::string control_address =
      server_info.cfg["control_ip"].as<std::string>() + ":" +
      server_info.cfg["control_port"].as<std::string>();
  FrankaTorqueControlClient franka_panda_client(server_info.server_ptr,
                                                server_info.cfg);
  franka_panda_client.run();

  return NULL;
}

int main(int argc, char **argv) {
  // Parse inputs
  InputParser input(argc, argv);

  if (input.cmdOptionExists("-h")) {
    spdlog::info("Usage: polymetis_server [OPTION]");
    spdlog::info("Starts a controller manager server.");
    spdlog::info("  -h   Help");
    spdlog::info("  -r   Use real-time (requires sudo)");
    spdlog::info("  -s   Change server address");
    return 0;
  }

  bool use_real_time = false;
  if (input.cmdOptionExists("-r")) {
    use_real_time = true;
  }

  std::string ip = "0.0.0.0";
  if (input.cmdOptionExists("-s")) {
    ip = input.getCmdOption("-s");
  }
  std::string port = "50051";
  if (input.cmdOptionExists("-p")) {
    port = input.getCmdOption("-p");
  }
  std::string server_address = ip + ":" + port;

  spdlog::info("Using real time: {}", use_real_time);
  spdlog::info("Using server address: {}", server_address);

  // Config
  std::string cfg_path = input.getCmdOption("-c");
  YAML::Node config = YAML::LoadFile(cfg_path);

  // Start server
  PolymetisControllerServerImpl service;
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());

  spdlog::info("Server listening on {}", server_address);

  // Start driver real-time thread
  ServerInfo server_info;
  server_info.server_ptr =
      std::shared_ptr<PolymetisControllerServerImpl>(&service);
  server_info.cfg = config;
  void *server_info_ptr = static_cast<void *>(&server_info);

  if (!use_real_time) {
    RunDriver(server_info_ptr);
  } else {
    create_real_time_thread(RunDriver, server_info_ptr);
  }

  // Wait for server to finish
  server->Wait();

  return 0;
}