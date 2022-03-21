// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <string>

#include <linux/can.h>
#include <linux/can/raw.h>

class PcanInterface {
public:
  explicit PcanInterface(std::string device_id);
  PcanInterface(PcanInterface &&pcan);
  PcanInterface &operator=(PcanInterface &&rhs) = delete;
  ~PcanInterface();

  void initialize();

  bool readPcan(can_frame *msg);
  bool writePcan(const can_frame &msg);

  static void printMsg(const can_frame &msg);

private:
  bool initialized_{false};
  int socket_;
  std::string device_id_;
};