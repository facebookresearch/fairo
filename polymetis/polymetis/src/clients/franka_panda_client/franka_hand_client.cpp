#include "polymetis/clients/franka_hand_client.hpp"

#include "spdlog/spdlog.h"
#include <string>
#include <thread>
#include <time.h>

#include <grpc/grpc.h>

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

FrankaHandClient::FrankaHandClient(std::shared_ptr<grpc::Channel> channel,
                                   YAML::Node config)
    : stub_(GripperServer::NewStub(channel)) {
  // Connect to gripper
  std::string robot_ip = config["robot_ip"].as<std::string>();
  spdlog::info("Connecting to robot_ip {}", robot_ip);
  gripper_.reset(new franka::Gripper(robot_ip));

  // Initialize gripper
  gripper_->homing();
  is_moving_ = false;
}

void FrankaHandClient::getGripperState(GripperState &gripper_state) {
  franka::GripperState franka_gripper_state = gripper_->readOnce();

  gripper_state.set_width(franka_gripper_state.width);
  gripper_state.set_max_width(franka_gripper_state.max_width);
  gripper_state.set_is_grasped(franka_gripper_state.is_grasped);
  gripper_state.set_is_moving(is_moving_);

  // gripper_state.time();  // Use current timestamp instead!
  setTimestampToNow(gripper_state.mutable_timestamp());
}

void FrankaHandClient::applyGripperCommand(GripperCommand &gripper_cmd) {
  if (gripper_cmd.grasp()) {
    spdlog::info("Grasping at width {} at speed={}", gripper_cmd.width(),
                 gripper_cmd.speed());
    is_moving_ = true;
    gripper_->grasp(gripper_cmd.width(), gripper_cmd.speed(),
                    gripper_cmd.force(), EPSILON_INNER, EPSILON_OUTER);
    is_moving_ = false;

  } else {
    spdlog::info("Moving to width {} at speed={}", gripper_cmd.width(),
                 gripper_cmd.speed());
    is_moving_ = true;
    gripper_->move(gripper_cmd.width(), gripper_cmd.speed());
    is_moving_ = false;
  }
}

void FrankaHandClient::run(void) {
  int period = 1.0 / GRIPPER_HZ;
  int period_ns = period * 1.0e9;

  struct timespec abs_target_time;
  clock_gettime(CLOCK_REALTIME, &abs_target_time);
  while (true) {
    // Run control step
    getGripperState(proto_gripper_state_);

    grpc::ClientContext context;
    status_ = stub_->ControlUpdate(&context, proto_gripper_state_,
                                   &proto_gripper_cmd_);

    if (!is_moving_) {
      int timestamp_ns = proto_gripper_cmd_.timestamp().nanos();

      // Skip if command not updated
      if (timestamp_ns != prev_cmd_timestamp_ns_) {
        // applyGripperCommand(proto_gripper_cmd_) in separate thread
        std::thread th(
            [this] { this->applyGripperCommand(proto_gripper_cmd_); });
        prev_cmd_timestamp_ns_ = timestamp_ns;
      }
    }

    // Spin once
    abs_target_time.tv_nsec += period_ns;
    clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &abs_target_time, nullptr);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: franka_hand_client /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  // Launch client
  std::string control_address = config["control_ip"].as<std::string>() + ":" +
                                config["control_port"].as<std::string>();
  FrankaHandClient franka_hand_client(
      grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
      config);
  franka_hand_client.run();

  // Termination
  spdlog::info("Wait for shutdown; press CTRL+C to close.");

  return 0;
}
