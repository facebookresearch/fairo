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
  if (robot_state_buffer_.size() == 0) {
    return Status(StatusCode::FAILED_PRECONDITION,
                  "Cannot retrieve robot state from empty buffer!");
  }
  *robot_state = *robot_state_buffer_.get(robot_state_buffer_.size() - 1);
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotClientMetadata(
    ServerContext *context, const Empty *, RobotClientMetadata *metadata) {
  if (!validRobotContext()) {
    return Status(
        StatusCode::CANCELLED,
        "Robot context not valid when calling GetRobotClientMetadata!");
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
  spdlog::info("==== Initializing new RobotClient... ====");

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
        controller_model_buffer_.data(), controller_model_buffer_.size(),
        *torch_robot_state_);
  } catch (const std::exception &e) {
    std::string error_msg =
        "Failed to load default controller: " + std::string(e.what());
    return Status(StatusCode::CANCELLED, error_msg);
  }

  // Set URDF file of new context
  robot_client_context_.metadata = RobotClientMetadata(*robot_client_metadata);

  // Set last updated timestep of robot client context
  robot_client_context_.last_update_ns = getNanoseconds();

  resetControllerContext();

  spdlog::info("Success.");
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

int PolymetisControllerServerImpl::setThreadPriority(int prio) {
  pthread_t curr_thr = pthread_self();
  int policy_noop;
  struct sched_param orig_param;

  pthread_getschedparam(curr_thr, &policy_noop, &orig_param);
  pthread_setschedprio(curr_thr, prio);

  // Return original prio for keeping track
  return orig_param.sched_priority;
}

Status
PolymetisControllerServerImpl::ControlUpdate(ServerContext *context,
                                             const RobotState *robot_state,
                                             TorqueCommand *torque_command) {
  // Check if last update is stale
  if (!validRobotContext()) {
    spdlog::warn("Interrupted control update greater than threshold of {} ns. "
                 "Reverting to default controller...",
                 threshold_ns_);
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

    spdlog::info(
        "Terminating custom controller, switching to default controller.");
  }

  // Select controller
  TorchScriptedController *controller;
  if (custom_controller_context_.status == RUNNING) {
    controller = custom_controller_context_.custom_controller.get();
  } else {
    controller = robot_client_context_.default_controller;
  }
  std::vector<float> desired_torque;
  try {
    desired_torque = controller->forward(*torch_robot_state_);
  } catch (const std::exception &e) {
    custom_controller_context_.controller_mtx.unlock();
    std::string error_msg =
        "Failed to run controller forward function: " + std::string(e.what());
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

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

  int orig_prio = setThreadPriority(RT_LOW_PRIO);

  interval->set_start(-1);
  interval->set_end(-1);

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
    auto new_controller = std::make_unique<TorchScriptedController>(
        controller_model_buffer_.data(), controller_model_buffer_.size(),
        *torch_robot_state_);

    // Switch in new controller by updating controller context
    // (note: use std::swap to put ptr to old controller in new_controller,
    // which destructs automatically after going out of scope)
    custom_controller_context_.controller_mtx.lock();

    resetControllerContext();
    std::swap(custom_controller_context_.custom_controller, new_controller);
    custom_controller_context_.status = READY;

    custom_controller_context_.controller_mtx.unlock();
    spdlog::info("Loaded new controller.");

  } catch (const std::exception &e) {
    std::string error_msg =
        "Failed to load new controller: " + std::string(e.what());
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  // Respond with start index
  while (custom_controller_context_.status == READY) {
    usleep(SPIN_INTERVAL_USEC);
  }
  interval->set_start(custom_controller_context_.episode_begin);

  setThreadPriority(orig_prio);
  return Status::OK;
}

Status PolymetisControllerServerImpl::UpdateController(
    ServerContext *context, ServerReader<ControllerChunk> *stream,
    LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);
  int orig_prio = setThreadPriority(RT_LOW_PRIO);

  interval->set_start(-1);
  interval->set_end(-1);

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
    std::string error_msg = "Failed to load new controller params.";
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  // Update controller & set intervals
  if (custom_controller_context_.status == RUNNING) {
    try {
      custom_controller_context_.controller_mtx.lock();
      interval->set_start(robot_state_buffer_.size());
      custom_controller_context_.custom_controller->param_dict_update_module();
      custom_controller_context_.controller_mtx.unlock();

    } catch (const std::exception &e) {
      custom_controller_context_.controller_mtx.unlock();

      std::string error_msg =
          "Failed to update controller: " + std::string(e.what());
      spdlog::error(error_msg);
      return Status(StatusCode::CANCELLED, error_msg);
    }

  } else {
    std::string error_msg =
        "Tried to perform a controller update with no controller running.";
    spdlog::warn(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  setThreadPriority(orig_prio);
  return Status::OK;
}

Status PolymetisControllerServerImpl::TerminateController(
    ServerContext *context, const Empty *, LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  interval->set_start(-1);
  interval->set_end(-1);

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
    std::string error_msg =
        "Tried to terminate controller with no controller running.";
    spdlog::warn(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  return Status::OK;
}

Status PolymetisControllerServerImpl::GetEpisodeInterval(
    ServerContext *context, const Empty *, LogInterval *interval) {
  interval->set_start(-1);
  interval->set_end(-1);

  if (custom_controller_context_.status != UNINITIALIZED) {
    interval->set_start(custom_controller_context_.episode_begin);
    interval->set_end(custom_controller_context_.episode_end);
  }

  return Status::OK;
}