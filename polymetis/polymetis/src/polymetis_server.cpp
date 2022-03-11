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
  controller_manager_.getRobotState(robot_state);
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotClientMetadata(
    ServerContext *context, const Empty *, RobotClientMetadata *metadata) {
  std::string error_msg;
  controller_manager_.GetRobotClientMetadata(metadata, error_msg);

  if (error_msg) {
    return Status(StatusCode::CANCELLED, error_msg);
  }
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
        controller_model_buffer_.data(), controller_model_buffer_.size());
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

Status
PolymetisControllerServerImpl::ControlUpdate(ServerContext *context,
                                             const RobotState *robot_state,
                                             TorqueCommand *torque_command) {
  std::string error_msg;
  controller_manager_.controlUpdate(robot_state, error_msg);

  if (error_msg) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  return Status::OK;
}

Status PolymetisControllerServerImpl::SetController(
    ServerContext *context, ServerReader<ControllerChunk> *stream,
    LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

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
    auto new_controller = new TorchScriptedController(
        controller_model_buffer_.data(), controller_model_buffer_.size());

    // Switch in new controller by updating controller context
    custom_controller_context_.controller_mtx.lock();

    resetControllerContext();
    custom_controller_context_.custom_controller = new_controller;
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

  // Return success.
  return Status::OK;
}

Status PolymetisControllerServerImpl::UpdateController(
    ServerContext *context, ServerReader<ControllerChunk> *stream,
    LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

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