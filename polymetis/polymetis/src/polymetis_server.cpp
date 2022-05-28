// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <string>

#include "polymetis/polymetis_server.hpp"

PolymetisControllerServerImpl::PolymetisControllerServerImpl() {
  controller_model_buffer_.reserve(MAX_MODEL_BYTES);
  updates_model_buffer_.reserve(MAX_MODEL_BYTES);
}

bool PolymetisControllerServerImpl::validRobotContext(void) {
  return controller_manager_.validRobotContext();
}

Status PolymetisControllerServerImpl::GetRobotState(ServerContext *context,
                                                    const Empty *,
                                                    RobotState *robot_state) {
  int latest_idx = controller_manager_.getStateBufferSize() - 1;
  *robot_state = *controller_manager_.getStateByBufferIndex(latest_idx);
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotClientMetadata(
    ServerContext *context, const Empty *, RobotClientMetadata *metadata) {
  std::string error_msg;
  controller_manager_.getRobotClientMetadata(metadata, error_msg);

  if (!error_msg.empty()) {
    return Status(StatusCode::CANCELLED, error_msg);
  }
  return Status::OK;
}

Status PolymetisControllerServerImpl::GetRobotStateStream(
    ServerContext *context, const Empty *, ServerWriter<RobotState> *writer) {
  uint i = controller_manager_.getStateBufferSize();
  while (!context->IsCancelled()) {
    if (i >= controller_manager_.getStateBufferSize()) {
      usleep(SPIN_INTERVAL_USEC);
    } else {
      RobotState *robot_state_ptr =
          controller_manager_.getStateByBufferIndex(i);
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
    end = controller_manager_.getStateBufferSize() - 1;
  }

  // Stream interval from robot state buffer
  for (uint i = interval->start(); i <= end; i++) {
    RobotState *robot_state_ptr = controller_manager_.getStateByBufferIndex(i);
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
  std::string error_msg;
  controller_manager_.initRobotClient(robot_client_metadata, error_msg);

  if (!error_msg.empty()) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  return Status::OK;
}

Status
PolymetisControllerServerImpl::ControlUpdate(ServerContext *context,
                                             const RobotState *robot_state,
                                             TorqueCommand *torque_command) {
  std::string error_msg;
  std::vector<float> desired_torque;

  // Query controller
  controller_manager_.controlUpdate(robot_state, desired_torque, error_msg);

  if (!error_msg.empty()) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  // Set output torque
  for (int i = 0; i < desired_torque.size(); i++) {
    torque_command->add_joint_torques(desired_torque[i]);
  }
  setTimestampToNow(torque_command->mutable_timestamp());

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

  // Set controller in controller manager
  std::string error_msg;
  controller_manager_.setController(controller_model_buffer_, interval,
                                    error_msg);

  if (!error_msg.empty()) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

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

  // Update controller in controller manager
  std::string error_msg;
  controller_manager_.updateController(updates_model_buffer_, interval,
                                       error_msg);

  if (!error_msg.empty()) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  return Status::OK;
}

Status PolymetisControllerServerImpl::TerminateController(
    ServerContext *context, const Empty *, LogInterval *interval) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  std::string error_msg;
  controller_manager_.terminateController(interval, error_msg);

  if (!error_msg.empty()) {
    spdlog::error(error_msg);
    return Status(StatusCode::CANCELLED, error_msg);
  }

  return Status::OK;
}

Status PolymetisControllerServerImpl::GetEpisodeInterval(
    ServerContext *context, const Empty *, LogInterval *interval) {
  controller_manager_.getEpisodeInterval(interval);

  return Status::OK;
}