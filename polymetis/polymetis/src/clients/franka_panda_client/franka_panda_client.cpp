// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "polymetis/clients/franka_panda_client.hpp"

#include "real_time.hpp"
#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <math.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#include <grpc/grpc.h>

using grpc::ClientContext;
using grpc::Status;

FrankaTorqueControlClient::FrankaTorqueControlClient(
    std::shared_ptr<grpc::Channel> channel, YAML::Node config)
    : stub_(PolymetisControllerServer::NewStub(channel)) {
  std::string robot_client_metadata_path =
      config["robot_client_metadata_path"].as<std::string>();

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
  mock_franka_ = config["mock"].as<bool>();
  readonly_mode_ = config["readonly"].as<bool>();
  if (!mock_franka_) {
    spdlog::info("Connecting to Franka Emika...");
    robot_ptr_.reset(new franka::Robot(config["robot_ip"].as<std::string>()));
    model_ptr_.reset(new franka::Model(robot_ptr_->loadModel()));
    spdlog::info("Connected.");
  } else {
    spdlog::info(
        "Launching Franka client in mock mode. No robot is connected.");
  }

  if (readonly_mode_) {
    spdlog::info("Launching Franka client in read only mode. No control will "
                 "be executed.");
  }

  // Set initial state & action
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_commanded_[i] = 0.0;
    torque_safety_[i] = 0.0;
    torque_applied_prev_[i] = 0.0;

    torque_command_.add_joint_torques(0.0);

    robot_state_.add_joint_positions(0.0);
    robot_state_.add_joint_velocities(0.0);
    robot_state_.add_prev_joint_torques_computed(0.0);
    robot_state_.add_prev_joint_torques_computed_safened(0.0);
    robot_state_.add_motor_torques_measured(0.0);
    robot_state_.add_motor_torques_external(0.0);
    robot_state_.add_motor_torques_desired(0.0);
  }

  // Parse yaml
  limit_rate_ = config["limit_rate"].as<bool>();
  lpf_cutoff_freq_ = config["lpf_cutoff_frequency"].as<double>();

  cartesian_pos_ulimits_ =
      config["limits"]["cartesian_pos_upper"].as<std::array<double, 3>>();
  cartesian_pos_llimits_ =
      config["limits"]["cartesian_pos_lower"].as<std::array<double, 3>>();
  joint_pos_ulimits_ =
      config["limits"]["joint_pos_upper"].as<std::array<double, NUM_DOFS>>();
  joint_pos_llimits_ =
      config["limits"]["joint_pos_lower"].as<std::array<double, NUM_DOFS>>();
  joint_vel_limits_ =
      config["limits"]["joint_vel"].as<std::array<double, NUM_DOFS>>();
  elbow_vel_limit_ = config["limits"]["elbow_vel"].as<double>();
  joint_torques_limits_ =
      config["limits"]["joint_torques"].as<std::array<double, NUM_DOFS>>();

  is_safety_controller_active_ =
      config["safety_controller"]["is_active"].as<bool>();

  margin_cartesian_pos_ =
      config["safety_controller"]["margins"]["cartesian_pos"].as<double>();
  margin_joint_pos_ =
      config["safety_controller"]["margins"]["joint_pos"].as<double>();
  margin_joint_vel_ =
      config["safety_controller"]["margins"]["joint_vel"].as<double>();

  k_cartesian_pos_ =
      config["safety_controller"]["stiffness"]["cartesian_pos"].as<double>();
  k_joint_pos_ =
      config["safety_controller"]["stiffness"]["joint_pos"].as<double>();
  k_joint_vel_ =
      config["safety_controller"]["stiffness"]["joint_vel"].as<double>();

  // Set collision behavior
  if (!mock_franka_) {
    robot_ptr_->setCollisionBehavior(
        config["collision_behavior"]["lower_torque"]
            .as<std::array<double, NUM_DOFS>>(),
        config["collision_behavior"]["upper_torque"]
            .as<std::array<double, NUM_DOFS>>(),
        config["collision_behavior"]["lower_force"].as<std::array<double, 6>>(),
        config["collision_behavior"]["upper_force"]
            .as<std::array<double, 6>>());
  }
}

void FrankaTorqueControlClient::run() {
  // Create callback function that relays information between gRPC server and
  // robot
  auto control_callback = [&](const franka::RobotState &libfranka_robot_state,
                              franka::Duration) -> franka::Torques {
    // Compute torque components
    updateServerCommand(libfranka_robot_state, torque_commanded_);
    checkStateLimits(libfranka_robot_state, torque_safety_);

    // Aggregate & clamp torques
    for (int i = 0; i < NUM_DOFS; i++) {
      torque_applied_[i] = torque_commanded_[i] + torque_safety_[i];
    }
    postprocessTorques(torque_applied_);

    // Record final applied torques
    for (int i = 0; i < NUM_DOFS; i++) {
      robot_state_.set_prev_joint_torques_computed_safened(i,
                                                           torque_applied_[i]);
    }

    return torque_applied_;
  };

  // Run robot
  if (!mock_franka_ && !readonly_mode_) {
    bool is_robot_operational = true;
    while (is_robot_operational) {
      // Send lambda function
      try {
        robot_ptr_->control(control_callback, limit_rate_, lpf_cutoff_freq_);
      } catch (const std::exception &ex) {
        spdlog::error("Robot is unable to be controlled: {}", ex.what());
        is_robot_operational = false;
      }

      // Automatic recovery
      spdlog::warn("Performing automatic error recovery. This calls "
                   "franka::Robot::automaticErrorRecovery, which is equivalent "
                   "to pressing and releasing the external activation device.");
      for (int i = 0; i < RECOVERY_MAX_TRIES; i++) {
        spdlog::warn("Automatic error recovery attempt {}/{}...", i + 1,
                     RECOVERY_MAX_TRIES);

        // Wait
        usleep(1000000 * RECOVERY_WAIT_SECS);

        // Attempt recovery
        try {
          robot_ptr_->automaticErrorRecovery();
          spdlog::warn("Robot operation recovered.");
          is_robot_operational = true;
          break;

        } catch (const std::exception &ex) {
          spdlog::error("Recovery failed: {}", ex.what());
        }
      }
    }

  } else {
    // Run mocked robot control loops
    franka::RobotState robot_state;
    franka::Duration duration;

    int period = 1.0 / FRANKA_HZ;
    int period_ns = period * 1.0e9;

    struct timespec abs_target_time;
    while (true) {
      clock_gettime(CLOCK_REALTIME, &abs_target_time);
      abs_target_time.tv_nsec += period_ns;

      // Pull data from robot if in readonly mode
      if (readonly_mode_) {
        robot_state = robot_ptr_->readOnce();
      }

      // Perform control loop with dummy variables (robot_state is populated if
      // in readonly mode)
      control_callback(robot_state, duration);

      clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &abs_target_time, nullptr);
    }
  }
}

void FrankaTorqueControlClient::updateServerCommand(
    /*
     * Send robot states and receive torque command via a request to the
     * controller server.
     */
    const franka::RobotState &libfranka_robot_state,
    std::array<double, NUM_DOFS> &torque_out) {
  // Record robot states
  if (!mock_franka_) {
    bool prev_command_successful = false;

    for (int i = 0; i < NUM_DOFS; i++) {
      robot_state_.set_joint_positions(i, libfranka_robot_state.q[i]);
      robot_state_.set_joint_velocities(i, libfranka_robot_state.dq[i]);
      robot_state_.set_motor_torques_measured(i,
                                              libfranka_robot_state.tau_J[i]);
      robot_state_.set_motor_torques_external(
          i, libfranka_robot_state.tau_ext_hat_filtered[i]);

      // Check if previous command is successful by checking whether
      // constant torque policy for packet drops is applied
      // (If applied, desired torques will be exactly the same as last timestep)
      if (!prev_command_successful &&
          float(libfranka_robot_state.tau_J_d[i]) !=
              robot_state_.motor_torques_desired(i)) {
        prev_command_successful = true;
      }
      robot_state_.set_motor_torques_desired(i,
                                             libfranka_robot_state.tau_J_d[i]);
    }

    robot_state_.set_prev_command_successful(prev_command_successful);

    // Error code: can only set to 0 if no errors and 1 if any errors exist for
    // now
    robot_state_.set_error_code(bool(libfranka_robot_state.current_errors));
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

void FrankaTorqueControlClient::checkStateLimits(
    const franka::RobotState &libfranka_robot_state,
    std::array<double, NUM_DOFS> &torque_out) {
  /*
   * Compute robot state limit violations and apply safety mechanisms.
   */
  std::array<double, 3> ee_pos_buf, force_buf;
  std::array<double, 1> elbow_vel_buf, elbow_lim_buf, dummy;

  // No safety checks in mock mode
  if (mock_franka_) {
    return;
  }

  // Reset reflex torques
  for (int i = 0; i < NUM_DOFS; i++) {
    torque_out[i] = 0.0;
  }
  for (int i = 0; i < 3; i++) {
    force_buf[i] = 0.0;
  }

  // Cartesian position limits
  for (int i = 0; i < 3; i++) {
    ee_pos_buf[i] = libfranka_robot_state.O_T_EE[12 + i];
  }
  computeSafetyReflex(ee_pos_buf, cartesian_pos_llimits_,
                      cartesian_pos_ulimits_, false, force_buf,
                      margin_cartesian_pos_, k_cartesian_pos_, "EE position");

  std::array<double, 6 *NUM_DOFS> jacobian_array = model_ptr_->zeroJacobian(
      franka::Frame::kEndEffector, libfranka_robot_state);
  Eigen::Map<const Eigen::Matrix<double, 6, NUM_DOFS>> jacobian(
      jacobian_array.data());
  Eigen::Map<const Eigen::Vector3d> force_xyz_vec(force_buf.data());

  Eigen::VectorXd force_vec(6);
  force_vec.head(3) << force_xyz_vec;
  force_vec.tail(3) << Eigen::Vector3d::Zero();

  Eigen::VectorXd torque_vec(NUM_DOFS);
  torque_vec << jacobian.transpose() * force_vec;
  Eigen::VectorXd::Map(&torque_out[0], NUM_DOFS) = torque_vec;

  // Joint position limits
  computeSafetyReflex(libfranka_robot_state.q, joint_pos_llimits_,
                      joint_pos_ulimits_, false, torque_out, margin_joint_pos_,
                      k_joint_pos_, "Joint position");

  // Joint velocity limits
  computeSafetyReflex(libfranka_robot_state.dq, joint_vel_limits_,
                      joint_vel_limits_, true, torque_out, margin_joint_vel_,
                      k_joint_vel_, "Joint velocity");

  // Miscellaneous velocity limits
  elbow_vel_buf[0] = libfranka_robot_state.delbow_c[0];
  elbow_lim_buf[0] = elbow_vel_limit_;
  computeSafetyReflex(elbow_vel_buf, elbow_lim_buf, elbow_lim_buf, true, dummy,
                      0.0, 0.0, "Elbow velocity");
}

void FrankaTorqueControlClient::postprocessTorques(
    /*
     * Filter & clamp torque to limits
     */
    std::array<double, NUM_DOFS> &torque_applied) {
  for (int i = 0; i < 7; i++) {
    // Clamp torques
    if (torque_applied[i] > joint_torques_limits_[i]) {
      torque_applied[i] = joint_torques_limits_[i];
    }
    if (torque_applied[i] < -joint_torques_limits_[i]) {
      torque_applied[i] = -joint_torques_limits_[i];
    }
  }
}

template <std::size_t N>
void FrankaTorqueControlClient::computeSafetyReflex(
    std::array<double, N> values, std::array<double, N> lower_limit,
    std::array<double, N> upper_limit, bool invert_lower,
    std::array<double, N> &safety_torques, double margin, double k,
    const char *item_name) {
  /*
   * Apply safety mechanisms for a vector based on input values and limits.
   * Throws an error if limits are violated.
   * Also computes & outputs safety controller torques.
   * (Note: invert_lower flips the sign of the lower limit. Used for velocities
   * and torques.)
   */
  double upper_violation, lower_violation;
  double lower_sign = 1.0;
  bool safety_constraint_triggered = false;

  std::string item_name_str(item_name);
  if (invert_lower) {
    lower_sign = -1.0;
  }

  // Init constraint active map
  if (active_constraints_map_.find(item_name_str) ==
      active_constraints_map_.end()) {
    active_constraints_map_.emplace(std::make_pair(item_name_str, false));
  }

  // Check limits & compute safety controller
  for (int i = 0; i < N; i++) {
    upper_violation = values[i] - upper_limit[i];
    lower_violation = lower_sign * lower_limit[i] - values[i];

    // Check hard limits (use active_constraints_map_ to prevent flooding
    // terminal)
    if (upper_violation > 0 || lower_violation > 0) {
      safety_constraint_triggered = true;

      if (!active_constraints_map_[item_name_str]) {
        active_constraints_map_[item_name_str] = true;

        spdlog::warn("Safety limits exceeded: "
                     "\n\ttype = \"{}\""
                     "\n\tdim = {}"
                     "\n\tlimits = {}, {}"
                     "\n\tvalue = {}",
                     item_name_str, i, lower_sign * lower_limit[i],
                     upper_limit[i], values[i]);

        std::string error_str =
            "Safety limits exceeded in FrankaTorqueControlClient. ";
        if (!readonly_mode_) {
          throw std::runtime_error(error_str + "\n");
        } else {
          spdlog::warn(error_str + "Ignoring issue during readonly mode.");
        }
      }
    }

    // Check soft limits & compute feedback forces (safety controller)
    if (is_safety_controller_active_) {
      if (upper_violation > -margin) {
        safety_torques[i] -= k * (margin + upper_violation);
      } else if (lower_violation > -margin) {
        safety_torques[i] += k * (margin + lower_violation);
      }
    }
  }

  // Reset constraint active map
  if (!safety_constraint_triggered && active_constraints_map_[item_name_str]) {
    active_constraints_map_[item_name_str] = false;
    spdlog::info("Safety limits no longer violated: \"{}\"", item_name_str);
  }
}

void *rt_main(void *cfg_ptr) {
  YAML::Node &config = *(static_cast<YAML::Node *>(cfg_ptr));

  // Launch adapter
  std::string control_address = config["control_ip"].as<std::string>() + ":" +
                                config["control_port"].as<std::string>();
  FrankaTorqueControlClient franka_panda_client(
      grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
      config);
  franka_panda_client.run();

  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: franka_panda_client /path/to/cfg.yaml");
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
