// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "controller_manager.hpp"

ControllerManager::ControllerManager(
    int num_dofs, std::vector<char> &default_controller_buffer) {
  num_dofs_ = num_dofs;
  default_controller_ = std::make_shared<TorchScriptedController>(
      default_controller_buffer.data(), default_controller_buffer.size());
}

// Interface methods

void ControllerManager::controlUpdate(TorchRobotState *torch_robot_state,
                                      std::vector<float> desired_torque,
                                      std::string &error_msg);
// Check if last update is stale
if (!validRobotContext()) {
  spdlog::warn("Interrupted control update greater than threshold of {} ns. "
               "Reverting to default controller...",
               threshold_ns_);
  controller_status_ = TERMINATING;
}

// Lock to prevent 1) controller updates while controller is running; 2)
// external termination during controller selection, which might cause loading
// of a uninitialized default controller
controller_mtx_.lock();

// Update episode markers
if (status_ == READY) {
  // First step of episode: update episode marker
  controller_status_ = RUNNING;

} else if (status_ == TERMINATING) {
  // Last step of episode: update episode marker & reset default controller
  controller_status_ = TERMINATED;
  default_controller_->reset();

  spdlog::info(
      "Terminating custom controller, switching to default controller.");
}

// Select controller
std::shared_ptr<TorchScriptedController> controller(nullptr);
if (custom_controller_context_.status == RUNNING) {
  controller = current_custom_controller_;
} else {
  controller = default_controller_;
}
std::vector<float> desired_torque;
try {
  desired_torque = controller->forward(*torch_robot_state_);
} catch (const std::exception &e) {
  controller_mtx_.unlock();
  error_msg =
      "Failed to run controller forward function: " + std::string(e.what());
  spdlog::error(error_msg);
  return;
}

// Unlock
controller_mtx_.unlock();
for (int i = 0; i < num_dofs_; i++) {
  torque_command->add_joint_torques(desired_torque[i]);
}
setTimestampToNow(torque_command->mutable_timestamp());

// Update timestep & check termination
if (controller_status_ == RUNNING) {
  if (controller->is_terminated()) {
    controller_status_ = TERMINATING;
  }
}
}

void ControllerManger::setController(std::string &error_msg) {
  try {
    // Load new controller
    auto new_controller = std::make_shared<TorchScriptedController>(
        controller_model_buffer_.data(), controller_model_buffer_.size());

    // Switch in new controller by updating controller context
    controller_mtx_.lock();

    controller_status_ = UNINITIALIZED current_custom_controller_ =
        new_controller;
    custom_controller_context_.status = READY;

    custom_controller_context_.controller_mtx.unlock();
    spdlog::info("Loaded new controller.");

  } catch (const std::exception &e) {
    error_msg = "Failed to load new controller: " + std::string(e.what());
    spdlog::error(error_msg);
    return;
  }
}

void updateController(std::vector<char> &update_buffer,
                      std::string &error_msg) {
  std::lock_guard<std::mutex> service_lock(service_mtx_);

  // Load param container
  if (!current_custom_controller_->param_dict_load(
          updates_model_buffer_.data(), updates_model_buffer_.size())) {
    error_msg = "Failed to load new controller params.";
    spdlog::error(error_msg);
    return;
  }

  // Update controller & set intervals
  if (controller_status_ == RUNNING) {
    try {
      controller_mtx_.lock();
      current_custom_controller_->param_dict_update_module();
      controller_mtx_.unlock();

    } catch (const std::exception &e) {
      controller_mtx_.unlock();

      error_msg = "Failed to update controller: " + std::string(e.what());
      spdlog::error(error_msg);
      return;
    }

  } else {
    error_msg =
        "Tried to perform a controller update with no controller running.";
    spdlog::warn(error_msg);
    return;
  }
}

void ControllerManger::terminateController(std::string &error_msg) {
  if (controller_status_ == RUNNING) {
    controller_mtx_.lock();
    controller_status_ = TERMINATING;
    controller_mtx_.unlock();

    // Respond with start & end index
    while (controller_status_ == TERMINATING) {
      usleep(SPIN_INTERVAL_USEC);
    }
    interval->set_start(custom_controller_context_.episode_begin);
    interval->set_end(custom_controller_context_.episode_end);

  } else {
    error_msg = "Tried to terminate controller with no controller running.";
    spdlog::warn(error_msg);
    return;
  }
}
