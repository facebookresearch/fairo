// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "./pcan_netdev_interface.hpp"
#include <fcntl.h>
#include <net/if.h>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

PcanInterface::PcanInterface(std::string device_id) : device_id_(device_id) {
  initialize();
}

PcanInterface::PcanInterface(PcanInterface &&pcan) {
  initialized_ = pcan.initialized_;
  pcan.initialized_ = false;
  socket_ = pcan.socket_;
  device_id_ = pcan.device_id_;
}

PcanInterface::~PcanInterface() {
  if (initialized_) {
    if (close(socket_) != 0) {
      spdlog::error("Error closing CAN bus socket: {}", strerror(errno));
    }
  }
}

// Adapted from [https://en.wikipedia.org/wiki/SocketCAN]
void PcanInterface::initialize() {
  sockaddr_can addr;
  ifreq ifr;

  if ((socket_ = socket(PF_CAN, SOCK_RAW, CAN_RAW)) == -1) {
    spdlog::warn("Failed to open CAN socket");
    throw std::runtime_error("Error opening CAN socket: " +
                             std::string(strerror(errno)));
  }

  strcpy(ifr.ifr_name, device_id_.c_str());
  if (ioctl(socket_, SIOCGIFINDEX, &ifr) == -1) {
    spdlog::warn("Trouble finding CAN bus {}: {}", device_id_, strerror(errno));
    throw std::runtime_error("Failed to initialize CAN interface");
  };

  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  spdlog::info("{} at index {}", device_id_.c_str(), ifr.ifr_ifindex);

  if (bind(socket_, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    throw std::runtime_error("Error binding CAN socket: " +
                             std::string(strerror(errno)));
  }

  // Set socket non-blocking
  int flags = fcntl(socket_, F_GETFL);
  if (flags == -1) {
    throw std::runtime_error("Error getting CAN socket flags: " +
                             std::string(strerror(errno)));
  }
  if (fcntl(socket_, F_SETFL, flags | O_NONBLOCK) != 0) {
    throw std::runtime_error("Error setting CAN socket non-blocking: " +
                             std::string(strerror(errno)));
  };
}

bool PcanInterface::readPcan(can_frame *msg) {
  int result = read(socket_, msg, sizeof(struct can_frame));

  if (result == -1) {
    if (errno != EAGAIN) {
      throw std::runtime_error("Error reading CAN socket:" +
                               std::string(strerror(errno)));
    }
    return false;
  }

  if (result != sizeof(can_frame)) {
    throw std::runtime_error("Read incomplete CAN frame.");
  }

  if (msg->can_id & CAN_ERR_FLAG) {
    printMsg(*msg);
    spdlog::error("Recieved CAN error.");
    return false;
  }
  return true;
}

bool PcanInterface::writePcan(const can_frame &msg) {
  int result = write(socket_, &msg, sizeof(can_frame));
  if (result != sizeof(can_frame)) {
    spdlog::warn("Failed to send CAN message: {}", strerror(errno));
    return false;
  }
  return true;
}

void PcanInterface::printMsg(const can_frame &msg) {
  spdlog::warn("msg: {:X}, {}", msg.can_id, msg.can_dlc);
}
