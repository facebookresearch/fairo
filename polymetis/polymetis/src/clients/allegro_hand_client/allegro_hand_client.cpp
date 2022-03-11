// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "polymetis/clients/allegro_hand_client.hpp"

#include "real_time.hpp"
#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"
#include <exception>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <fstream>

#include <grpc/grpc.h>

#include "./allegro_hand.hpp"
#include "./event_watcher.hpp"
#include "./finger_velocity_filter.hpp"

using grpc::ClientContext;
using grpc::Status;

struct AllegroHandState {
  std::array<double, kNDofs> q, dq;
};

AllegroHandTorqueControlClient::AllegroHandTorqueControlClient(
    std::shared_ptr<grpc::Channel> channel, YAML::Node config)
    : stub_(PolymetisControllerServer::NewStub(channel)) {
  std::string robot_client_metadata_path =
      config["robot_client_metadata_path"].as<std::string>();

  velocity_filter_ =
      std::make_unique<FingerVelocityFilter>(config["velocity_filter"]);

  // Load robot client metadata
  std::ifstream file(robot_client_metadata_path);
  assert(file);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  RobotClientMetadata metadata;
  assert(metadata.ParseFromString(buffer.str()));

  // Initialize robot client with metadata
  ClientContext context;
  Empty empty;
  Status status = stub_->InitRobotClient(&context, metadata, &empty);
  assert(status.ok());

  // Connect to robot
  mock_allegro_ = config["mock"].as<bool>();
  readonly_mode_ = config["readonly"].as<bool>();
  if (!mock_allegro_) {
    spdlog::info("Connecting to Allegro Hand...");

    const std::string can_bus_id = config["can_bus"].as<std::string>();
    allegro_hand_ptr_.reset(new AllegroHandImpl(PcanInterface(can_bus_id)));
    spdlog::info("Connected.");
  } else {
    spdlog::info(
        "Launching Allegro Hand client in mock mode. No robot is connected.");
    allegro_hand_ptr_.reset(new MockAllegroHand());
  }

  if (readonly_mode_) {
    spdlog::info(
        "Launching Allegro Hand client in read only mode. No control will "
        "be executed.");
  }

  // Set initial state & action
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_commanded_[i] = 0.0;
    torque_safety_[i] = 0.0;

    torque_command_.add_joint_torques(0.0);

    robot_state_.add_joint_positions(0.0);
    robot_state_.add_joint_velocities(0.0);
    robot_state_.add_prev_joint_torques_computed(0.0);
    robot_state_.add_prev_joint_torques_computed_safened(0.0);
    robot_state_.add_motor_torques_measured(0.0);
    robot_state_.add_motor_torques_external(0.0);
  }
}

void AllegroHandTorqueControlClient::run() {
  // Run robot
  EventData poll_success_event;
  AllegroHandState robot_state;

  int recovery_attempts = 0;
  allegro_hand_ptr_->setServoEnable(true);
  allegro_hand_ptr_->requestStatus();
  PeriodicEvent statdump_event(60);
  while (recovery_attempts < RECOVERY_MAX_TRIES) {
    try {
      while (true) {
        if (statdump_event) {
          std::ostringstream log_writer;
          log_writer << std::endl << gEventWatcher;
          spdlog::info(log_writer.str());
        }
        while (!allegro_hand_ptr_->allStatesUpdated()) {
          int finger = allegro_hand_ptr_->poll();
          if (finger >= 0) {
            std::copy_n(allegro_hand_ptr_->getPositions(), kNDofs,
                        robot_state.q.begin());
            Eigen::Map<Eigen::VectorXd> positions(robot_state.q.data(), kNDofs);
            Eigen::Map<Eigen::VectorXd> velocities(robot_state.dq.data(),
                                                   kNDofs);
            velocity_filter_->filter_position(positions, finger, velocities);
            poll_success_event.checkpoint();
            recovery_attempts = 0; // Comms are working / restored.
          } else if (poll_success_event.secondsSinceCheckpoint() > 3) {
            throw std::runtime_error("No recent communication from hand.");
          }
        }
        allegro_hand_ptr_->resetStateUpdateTracker();

        // Compute torque components
        updateServerCommand(robot_state, torque_commanded_);

        allegro_hand_ptr_->setTorques(torque_commanded_.begin());
      }
    } catch (const std::exception &ex) {
      spdlog::error("Robot is unable to be controlled: {}", ex.what());
    }

    // Attempt recovery
    spdlog::warn("Attempting to reconnect to Allegro Hand");
    allegro_hand_ptr_->resetCommunication();
    allegro_hand_ptr_->setServoEnable(true);
    allegro_hand_ptr_->requestStatus();
    poll_success_event.checkpoint(); // reset the comm heartbeat timer

    recovery_attempts++;
  }
  spdlog::error("Max recovery attempts exhausted.");
}

void AllegroHandTorqueControlClient::updateServerCommand(
    /*
     * Send robot states and receive torque command via a request to the
     * controller server.
     */
    const AllegroHandState &robot_state,
    std::array<double, NUM_DOFS> &torque_out) {
  // Record robot states
  for (int i = 0; i < NUM_DOFS; i++) {
    robot_state_.set_joint_positions(i, robot_state.q[i]);
    robot_state_.set_joint_velocities(i, robot_state.dq[i]);
  }
  setTimestampToNow(robot_state_.mutable_timestamp());

  // Retrieve torques
  grpc::ClientContext context;
  long int pre_update_ns = getNanoseconds();
  status_ = stub_->ControlUpdate(&context, robot_state_, &torque_command_);
  long int post_update_ns = getNanoseconds();
  if (!status_.ok()) {
    std::string error_msg = "ControlUpdate rpc failed. ";
    throw std::runtime_error(error_msg + status_.error_message());
  }

  robot_state_.set_prev_controller_latency_ms(
      float(post_update_ns - pre_update_ns) / 1e6);

  assert(torque_command_.joint_torques_size() == NUM_DOFS);
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_out[i] = torque_command_.joint_torques(i);
    robot_state_.set_prev_joint_torques_computed(
        i, torque_command_.joint_torques(i));
  }
}

void *rt_main(void *cfg_ptr) {
  YAML::Node &config = *(static_cast<YAML::Node *>(cfg_ptr));

  // Launch adapter
  std::string control_address;
  try {
    control_address = config["control_ip"].as<std::string>() + ":" +
                      config["control_port"].as<std::string>();
  } catch (...) {
    std::throw_with_nested(std::runtime_error(
        "Failed to read Polymetis server address from config."));
  }
  AllegroHandTorqueControlClient allegro_hand_client(
      grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
      config);
  allegro_hand_client.run();

  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: allegro_hand_client /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);
  void *config_void_ptr = static_cast<void *>(&config);

  // Launch thread
  create_real_time_thread(rt_main, config_void_ptr);

  // Termination
  spdlog::info("Wait for shutdown; press CTRL+C to close.");

  return 0;
}
