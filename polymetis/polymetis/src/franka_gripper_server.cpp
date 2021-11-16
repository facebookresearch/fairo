#include "yaml-cpp/yaml.h"
#include <iostream>

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

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

// Define tolerances to be able to grasp any object without specifying width
#define EPSILON_INNER -0.0005
#define EPSILON_OUTER 0.15

class GripperControllerImpl final : public GripperServer::Service {
public:
  explicit GripperControllerImpl(std::string robot_ip) {
    std::cout << "Connecting to robot_ip " << robot_ip << std::endl;
    gripper_ = new franka::Gripper(robot_ip);
    gripper_state_ = new franka::GripperState;
    is_moving_ = false;

    // Initialize gripper
    gripper_->homing();
  }

  Status GetState(ServerContext *context, const Empty *,
                  GripperState *gripper_proto_state) override {
    *gripper_state_ = gripper_->readOnce();
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
    std::cout << "Moving to width " << gripper_command->width()
              << " at speed=" << gripper_command->speed() << std::endl;
    is_moving_ = true;
    gripper_->stop();
    gripper_->move(gripper_command->width(), gripper_command->speed());
    is_moving_ = false;

    return Status::OK;
  }

  Status Grasp(ServerContext *context, const GripperCommand *gripper_command,
               Empty *) override {
    is_moving_ = true;
    gripper_->stop();
    gripper_->grasp(gripper_command->width(), gripper_command->speed(),
                    gripper_command->force(), EPSILON_INNER, EPSILON_OUTER);
    is_moving_ = false;

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
    std::cout << "Usage: ./franka_gripper_server /path/to/cfg.yaml"
              << std::endl;
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
  std::cout << "Franka Hand server listening on " << server_address
            << std::endl;
  server->Wait();

  return 0;
}