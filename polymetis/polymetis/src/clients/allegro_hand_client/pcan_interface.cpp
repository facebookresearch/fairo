// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "pcan_interface.hpp"

#include <cstdio>
#include <stdexcept>

#include <spdlog/spdlog.h>

void PcanInterface::printError(TPCANStatus status) {
  char err_msg[256];
  CAN_GetErrorText(status, 0, err_msg);
  spdlog::error("CAN Bus Error: {}", err_msg);
}

void PcanInterface::printMsg(const TPCANMsg &msg,
                             const TPCANTimestamp &timestamp) {
  spdlog::info("{12u}:{06u} ID:{x} TYPE:{x}", timestamp.millis,
               timestamp.micros, msg.ID >> 2, msg.MSGTYPE);
}

PcanInterface::PcanInterface(TPCANHandle bus_id) : bus_id_(bus_id) {
  initialize();
}

void PcanInterface::initialize() {
  if (initialized_) {
    CAN_Uninitialize(bus_id_);
  }
  TPCANStatus pcan_status = CAN_Initialize(bus_id_, PCAN_BAUD_1M, 0);
  if (pcan_status != PCAN_ERROR_OK) {
    printError(pcan_status);
    throw std::runtime_error(
        "PCAN initialization failed.  Perhaps you need to 'modprobe pcan'?");
  }
  initialized_ = true;
}

PcanInterface::PcanInterface(PcanInterface &&pcan) : bus_id_(pcan.bus_id_) {
  initialized_ = pcan.initialized_;
  pcan.initialized_ = false;
}

bool PcanInterface::readPcan(TPCANMsg *msg) {
  TPCANTimestamp timestamp;
  TPCANStatus status = CAN_Read(bus_id_, msg, &timestamp);

  if (status != PCAN_ERROR_OK) {
    if (status != PCAN_ERROR_QRCVEMPTY) {
      printError(status);
    }
    return false;
  }
  return true;
}

bool PcanInterface::writePcan(const TPCANMsg &msg) {
  TPCANStatus status;
  if (msg.LEN > 8) {
    spdlog::error("Bad MSG Len: tried to write a message with length {} (max "
                  "data len is 8 bytes).",
                  msg.LEN);
    return false;
  }

  status = CAN_Write(bus_id_, const_cast<TPCANMsg *>(&msg));
  TPCANTimestamp ts;
  if (status != PCAN_ERROR_OK) {
    printError(status);
    return false;
  }
  return true;
}

PcanInterface::~PcanInterface() {
  if (initialized_) {
    CAN_Uninitialize(bus_id_);
  }
}
