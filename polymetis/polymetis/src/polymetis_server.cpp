// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <string>

#include "polymetis/polymetis_server.hpp"

PolymetisControllerServerImpl::PolymetisControllerServerImpl() {
  controller_model_buffer_.reserve(MAX_MODEL_BYTES);
  updates_model_buffer_.reserve(MAX_MODEL_BYTES);
}

Status PolymetisControllerServerImpl::GetRobotState(ServerContext *context,
                                                    const Empty *,
                                                    RobotState *robot_state) {
  *robot_state = *robot_state_buffer_.get(robot_state_buffer_.size() - 1);
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotClientMetadata(
    ServerContext *context, const Empty *, RobotClientMetadata *metadata) {
  if (!validRobotContext()) {
    return Status::CANCELLED;
  }
  *metadata = robot_client_context_.metadata;
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotStateStream(
    ServerContext *context, const Empty *, ServerWriter<RobotState> *writer) {
  uint i = robot_state_buffer_.size();
  while (!context->IsCancelled()) {
    if (i >= robot_state_buffer_.size()) {
      usleep(SPIN_INTERVAL_USEC);
    } else {
      RobotState *robot_state_ptr = robot_state_buffer_.get(i);
      if (robot_state_ptr != NULL) {
        writer->Write(RobotState(*robot_state_ptr));
      }
      i++;
    }
  }
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotStateLog(
    ServerContext *context, const LogInterval *interval,
    ServerWriter<RobotState> *writer) {
  // Stream until latest if end == -1
  uint end = interval->end();
  if (interval->end() == -1) {
    end = robot_state_buffer_.size() - 1;
  }

  // Stream interval from robot state buffer
  for (uint i = interval->start(); i <= end; i++) {
    RobotState *robot_state_ptr = robot_state_buffer_.get(i);
    if (robot_state_ptr != NULL) {
      writer->Write(RobotState(*robot_state_ptr));
    }

    // Break if request cancelled
    if (context->IsCancelled()) {
      break;
    }
  }
  return Status::OK;
}

Status PolymetisControllerServerImpl::InitRobotClient(
    ServerContext *context, const RobotClientMetadata *robot_client_metadata,
    Empty *) {
  std::cout << "\n\n==== Initializing new RobotClient... ====" << std::endl;

  num_dofs_ = robot_client_metadata->dof();

  torch_robot_state_ =
      std::unique_ptr<TorchRobotState>(new TorchRobotState(num_dofs_));

  // Load default controller bytes into model buffer
  controller_model_buffer_.clear();
  std::string binary_blob = robot_client_metadata->default_controller();
  for (int i = 0; i < binary_blob.size(); i++) {
    controller_model_buffer_.push_back(binary_blob[i]);
  }

  // Load default controller from model buffer
  try {
    robot_client_context_.default_controller = new TorchScriptedController(
        controller_model_buffer_.data(), controller_model_buffer_.size());
  } catch (const std::exception &e) {
    std::cerr << "error loading default controller:\n";
    std::cerr << e.what() << std::endl;
    return Status::CANCELLED;
  }

  // Set URDF file of new context
  robot_client_context_.metadata = RobotClientMetadata(*robot_client_metadata);

  // Set last updated timestep of robot client context
  robot_client_context_.last_update_ns = getNanoseconds();

  resetControllerContext();

  std::cout << "Success.\n\n";
  return Status::OK;
}

bool PolymetisControllerServerImpl::validRobotContext() {
  if (robot_client_context_.last_update_ns == 0) {
    return false;
  }
  long int time_since_last_update =
      getNanoseconds() - robot_client_context_.last_update_ns;
  return time_since_last_update < threshold_ns_;
}

void PolymetisControllerServerImpl::resetControllerContext() {
  custom_controller_context_.episode_begin = -1;
  custom_controller_context_.episode_end = -1;
  custom_controller_context_.timestep = 0;
  custom_controller_context_.status = UNINITIALIZED;
}

Status
PolymetisControllerServerImpl::ControlUpdate(ServerContext *context,
                                             const RobotState *robot_state,
                                             TorqueCommand *torque_command) {
  auto start = std::chrono::high_resolution_clock::now();

  // Check if last update is stale
  if (!validRobotContext()) {
    std::cout
        << "Warning: Interrupted control update greater than threshold of "
        << threshold_ns_ << " ns. Reverting to default controller..."
        << std::endl;
    custom_controller_context_.status = TERMINATING;
  }

  // Parse robot state
  torch_robot_state_->update_state(
      robot_state->timestamp().seconds(), robot_state->timestamp().nanos(),
      std::vector<float>(robot_state->joint_positions().begin(),
                         robot_state->joint_positions().end()),
      std::vector<float>(robot_state->joint_velocities().begin(),
                         robot_state->joint_velocities().end()),
      std::vector<float>(robot_state->motor_torques_measured().begin(),
                         robot_state->motor_torques_measured().end()),
      std::vector<float>(robot_state->motor_torques_external().begin(),
                         robot_state->motor_torques_external().end()));

  // Lock to prevent 1) controller updates while controller is running; 2)
  // external termination during controller selection, which might cause loading
  // of a uninitialized default controller
  custom_controller_context_.controller_mtx.lock();

  // Update episode markers
  if (custom_controller_context_.status == READY) {
    // First step of episode: update episode marker
    custom_controller_context_.episode_begin = robot_state_buffer_.size();
    custom_controller_context_.status = RUNNING;

  } else if (custom_controller_context_.status == TERMINATING) {
    // Last step of episode: update episode marker & reset default controller
    custom_controller_context_.episode_end = robot_state_buffer_.size() - 1;
    custom_controller_context_.status = TERMINATED;

    robot_client_context_.default_controller->reset();

    std::cout
        << "Terminating custom controller, switching to default controller."
        << std::endl;
  }

  // Select controller
  TorchScriptedController *controller;
  if (custom_controller_context_.status == RUNNING) {
    controller = custom_controller_context_.custom_controller;
  } else {
    controller = robot_client_context_.default_controller;
  }

  std::vector<float> desired_torque = controller->forward(*torch_robot_state_);
  // Unlock
  custom_controller_context_.controller_mtx.unlock();
  for (int i = 0; i < num_dofs_; i++) {
    torque_command->add_joint_torques(desired_torque[i]);
  }
  setTimestampToNow(torque_command->mutable_timestamp());

  // Record robot state
  RobotState robot_state_copy(*robot_state);
  for (int i = 0; i < num_dofs_; i++) {
    robot_state_copy.add_joint_torques_computed(
        torque_command->joint_torques(i));
  }

  // Record loop time
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  robot_state_copy.set_control_loop_ms(double(duration.count()) / 1000.0);

  // Append robot state
  robot_state_buffer_.append(robot_state_copy);

  // Update timestep & check termination
  if (custom_controller_context_.status == RUNNING) {
    custom_controller_context_.timestep++;
    if (controller->is_terminated()) {
      custom_controller_context_.status = TERMINATING;
    }
  }

  robot_client_context_.last_update_ns = getNanoseconds();

  return Status::OK;
}

Status PolymetisControllerServerImpl::SetController(
    ServerContext *context, ServerReader<ControllerChunk> *stream,
    LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  // Read chunks of the binary serialized controller. The binary messages
  // would be written into the preallocated buffer used for the Torch
  // controllers.
  controller_model_buffer_.clear();
  ControllerChunk chunk;
  while (stream->Read(&chunk)) {
    std::string binary_blob = chunk.torchscript_binary_chunk();
    for (int i = 0; i < binary_blob.size(); i++) {
      controller_model_buffer_.push_back(binary_blob[i]);
    }
  }

  try {
    // Load new controller
    auto new_controller = new TorchScriptedController(
        controller_model_buffer_.data(), controller_model_buffer_.size());

    // Switch in new controller by updating controller context
    custom_controller_context_.controller_mtx.lock();

    resetControllerContext();
    custom_controller_context_.custom_controller = new_controller;
    custom_controller_context_.status = READY;

    custom_controller_context_.controller_mtx.unlock();
    std::cout << "Loaded new controller.\n";

  } catch (const std::exception &e) {
    std::cerr << "error loading the model:\n";
    std::cerr << e.what() << std::endl;

    return Status::CANCELLED;
  }

  // Respond with start index
  while (custom_controller_context_.status == READY) {
    usleep(SPIN_INTERVAL_USEC);
  }
  interval->set_start(custom_controller_context_.episode_begin);
  interval->set_end(-1);

  // Return success.
  return Status::OK;
}

Status PolymetisControllerServerImpl::UpdateController(
    ServerContext *context, ServerReader<ControllerChunk> *stream,
    LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  // Read chunks of the binary serialized controller params container.
  updates_model_buffer_.clear();
  ControllerChunk chunk;
  while (stream->Read(&chunk)) {
    std::string binary_blob = chunk.torchscript_binary_chunk();
    for (int i = 0; i < binary_blob.size(); i++) {
      updates_model_buffer_.push_back(binary_blob[i]);
    }
  }

  // Load param container
  if (!custom_controller_context_.custom_controller->param_dict_load(
          updates_model_buffer_.data(), updates_model_buffer_.size())) {
    return Status::CANCELLED;
  }

  // Update controller & set intervals
  if (custom_controller_context_.status == RUNNING) {
    custom_controller_context_.controller_mtx.lock();
    interval->set_start(robot_state_buffer_.size());
    custom_controller_context_.custom_controller->param_dict_update_module();
    custom_controller_context_.controller_mtx.unlock();
  } else {
    std::cout << "Warning: Tried to perform a controller update with no "
                 "controller running.\n";
    interval->set_start(-1);
  }

  interval->set_end(-1);

  return Status::OK;
}

Status PolymetisControllerServerImpl::TerminateController(
    ServerContext *context, const Empty *, LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  if (custom_controller_context_.status == RUNNING) {
    custom_controller_context_.controller_mtx.lock();
    custom_controller_context_.status = TERMINATING;
    custom_controller_context_.controller_mtx.unlock();

    // Respond with start & end index
    while (custom_controller_context_.status == TERMINATING) {
      usleep(SPIN_INTERVAL_USEC);
    }
    interval->set_start(custom_controller_context_.episode_begin);
    interval->set_end(custom_controller_context_.episode_end);
  } else {
    std::cout << "Warning: Tried to terminate controller with no controller "
                 "running.\n";
    interval->set_start(-1);
    interval->set_end(-1);
  }

  return Status::OK;
}

Status PolymetisControllerServerImpl::GetEpisodeInterval(
    ServerContext *context, const Empty *, LogInterval *interval) {
  if (custom_controller_context_.status != UNINITIALIZED) {
    interval->set_start(custom_controller_context_.episode_begin);
    interval->set_end(custom_controller_context_.episode_end);
  } else {
    interval->set_start(-1);
    interval->set_end(-1);
  }

  return Status::OK;
}