#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::Status;
using grpc::StatusCode;

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

// Define tolerances to be able to grasp any object without specifying width
#define EPSILON_INNER -0.0005
#define EPSILON_OUTER 0.15

class GripperControllerImpl final : public GripperServer::Service {
public:
  explicit GripperControllerImpl(std::string robot_ip) {
    spdlog::info("Connecting to robot_ip {}", robot_ip);
    gripper_ = new franka::Gripper(robot_ip);
    gripper_state_ = new franka::GripperState;
    is_moving_ = false;

    // Initialize gripper
    gripper_->homing();
  }

  Status GetState(ServerContext *context, const Empty *,
                  GripperState *gripper_proto_state) override {
    // Read gripper state
    try {
      *gripper_state_ = gripper_->readOnce();
    } catch (const std::exception &e) {
      std::string error_msg =
          "Failed to read from Franka Hand: " + std::string(e.what());
      return Status(StatusCode::CANCELLED, error_msg);
    }

    // Construct gripper state msg
    gripper_proto_state->set_width(gripper_state_->width);
    gripper_proto_state->set_max_width(gripper_state_->max_width);
    gripper_proto_state->set_is_grasped(gripper_state_->is_grasped);
    gripper_proto_state->set_is_moving(is_moving_);

    // gripper_state_->time();  // Use current timestamp instead!
    setTimestampToNow(gripper_proto_state->mutable_timestamp());

    return Status::OK;
  }

  Status Goto(ServerContext *context, const GripperCommand *gripper_command,
              Empty *) override {
    spdlog::info("Moving to width {} at speed={}", gripper_command->width(),
                 gripper_command->speed());

    try {
      is_moving_ = true;
      gripper_->move(gripper_command->width(), gripper_command->speed());
      is_moving_ = false;

    } catch (const std::exception &e) {
      std::string error_msg =
          "Failed to command Franka Hand: " + std::string(e.what());
      return Status(StatusCode::CANCELLED, error_msg);
    }

    return Status::OK;
  }

  Status Grasp(ServerContext *context, const GripperCommand *gripper_command,
               Empty *) override {
    spdlog::info("Grasping at width {} at speed={}", gripper_command->width(),
                 gripper_command->speed());

    try {
      is_moving_ = true;
      gripper_->grasp(gripper_command->width(), gripper_command->speed(),
                      gripper_command->force(), EPSILON_INNER, EPSILON_OUTER);
      is_moving_ = false;

    } catch (const std::exception &e) {
      std::string error_msg =
          "Failed to command Franka Hand: " + std::string(e.what());
      return Status(StatusCode::CANCELLED, error_msg);
    }

    return Status::OK;
  }

private:
  franka::Gripper *gripper_;
  franka::GripperState *gripper_state_;
  bool is_moving_;
};

int main(int argc, char *argv[]) {
  // Parse config
  if (argc != 2) {
    spdlog::error("Usage: ./franka_gripper_server /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  // Create service
  GripperControllerImpl service(config["robot_ip"].as<std::string>());

  // Launch service
  std::string server_address =
      config["ip"].as<std::string>() + ":" + config["port"].as<std::string>();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  spdlog::info("Franka Hand server listening on {}", server_address);
  server->Wait();

  return 0;
}