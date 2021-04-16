// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "types.h"

const int PROTO_VERSION = 340;
const int THRESHOLD = 256;

class Encoder {
 public:
  Encoder();
  void clear();
  void print();
  static std::vector<uint8_t> _varint(long x);

  // types
  void bytes(std::vector<uint8_t> bs);
  void varint(long x);
  void str(std::string x);
  void byte(uint8_t x) { bigEndian(x, 1); }
  void boolean(bool x) { bigEndian(x, 1); }
  void uint16(uint16_t x) { bigEndian(x, 2); }
  void int32(int32_t x) { bigEndian(x, 4); }
  void int64(int64_t x) { bigEndian(x, 8); }
  void uint64(uint64_t x) { bigEndian(x, 8); }
  void float32(float x);
  void float64(double x);
  void position(int64_t x, int64_t y, int64_t z);
  void position(BlockPos p) { position(p.x, p.y, p.z); }
  void slot(Slot s);

  // packets
  std::vector<uint8_t> packet(int pid, size_t threshold);
  std::vector<uint8_t> handshakePacket(const std::string& host, int port, bool status);
  std::vector<uint8_t> loginStartPacket(const std::string& username);
  std::vector<uint8_t> chatMessagePacket(const std::string& message);
  std::vector<uint8_t> playerPositionPacket(double x, double y, double z, bool onGround);
  std::vector<uint8_t> teleportConfirmPacket(long teleportId);
  std::vector<uint8_t> keepalivePacket(long keepaliveId);
  std::vector<uint8_t> heldItemChangePacket(uint8_t i);
  std::vector<uint8_t> creativeInventoryActionPacket(int16_t index, Slot slot);
  std::vector<uint8_t> playerBlockPlacementPacket(BlockPos p);
  std::vector<uint8_t> playerUseEntityPacket(BlockPos p);
  std::vector<uint8_t> playerUseItemPacket();
  std::vector<uint8_t> playerLookPacket(float yaw, float pitch, bool onGround);
  std::vector<uint8_t> playerStartDiggingPacket(BlockPos pos);
  std::vector<uint8_t> playerFinishedDiggingPacket(BlockPos pos);
  std::vector<uint8_t> playerDropItemStackInHandPacket();
  std::vector<uint8_t> playerDropItemInHandPacket();
  std::vector<uint8_t> playerDropItemStackPacket(Slot slot);
  std::vector<uint8_t> playerSetInventorySlotPacket(int16_t index, Slot slot);

  std::vector<uint8_t> clickWindowPacket(uint8_t windowId, uint16_t slot, bool rightClick,
                                         uint16_t counter, Slot clicked);
  std::vector<uint8_t> closeWindowPacket(uint8_t windowId);

 private:
  void bigEndian(unsigned long x, size_t n);
  static void append(std::vector<uint8_t>& to, std::vector<uint8_t> from) {
    to.insert(to.end(), from.begin(), from.end());
  }

  // private fields
  std::vector<uint8_t> buf_;
};
