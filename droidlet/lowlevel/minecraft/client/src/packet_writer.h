// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <mutex>
#include <vector>

class PacketWriter {
 public:
  PacketWriter() {}
  PacketWriter(int s) : socket_(s) {}
  void safeWrite(std::vector<uint8_t> bs);

 private:
  void socketSend(std::vector<uint8_t> bs);

  // fields
  int socket_;
  std::mutex lock_;
};
