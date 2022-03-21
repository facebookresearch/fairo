// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <string>

#include "PCANBasic.h"

class PcanInterface {
public:
  explicit PcanInterface(TPCANHandle bus_id);
  PcanInterface(PcanInterface &&pcan);
  PcanInterface &operator=(PcanInterface &&rhs) = delete;
  ~PcanInterface();

  void initialize();

  bool readPcan(TPCANMsg *msg);
  bool writePcan(const TPCANMsg &msg);

  static void printError(TPCANStatus status);
  static void printMsg(const TPCANMsg &msg, const TPCANTimestamp &timestamp);

private:
  bool initialized_{false};
  const TPCANHandle bus_id_;
};
