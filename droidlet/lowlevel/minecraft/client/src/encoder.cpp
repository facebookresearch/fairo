// Copyright (c) Facebook, Inc. and its affiliates.

#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "encoder.h"

using namespace std;

////////////////
// Public

Encoder::Encoder() {
  buf_.reserve(1 << 16);  // 64k buffer
}

void Encoder::clear() { buf_.clear(); }

void Encoder::print() {
  cout << "0x";
  for (auto b : buf_) {
    cout << setw(2) << setfill('0') << hex << (int)b;
  }
  cout << '\n';
}

void Encoder::bytes(vector<uint8_t> bs) { append(buf_, bs); }

// See http://wiki.vg/Protocol#VarInt_and_VarLong
void Encoder::varint(long x) { bytes(_varint(x)); }

void Encoder::str(string x) {
  varint(x.size());
  for (auto c : x) buf_.push_back(c);
}

void Encoder::float32(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  bigEndian(u, 4);
}

void Encoder::float64(double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  bigEndian(u, 8);
}

void Encoder::position(int64_t x, int64_t y, int64_t z) {
  long v = ((x & 0x3ffffff) << 38) | ((y & 0xfff) << 26) | (z & 0x3ffffff);
  uint64(v);
}

void Encoder::slot(Slot s) {
  uint16(s.id);
  byte(s.count ? s.count : 1);
  uint16(s.meta);  // supposed to be damage, but seems to hold meta???
  byte(0);         // no nbt data
}

// Packetize and return the contents of buf_
vector<uint8_t> Encoder::packet(int pid, size_t threshold) {
  vector<uint8_t> out;
  vector<uint8_t> enc_pid = _varint(pid);
  if (threshold == 0) {
    append(out, _varint(enc_pid.size() + buf_.size()));
    append(out, enc_pid);
    append(out, buf_);
  } else if (enc_pid.size() + buf_.size() < threshold) {
    auto data_len = _varint(0);
    append(out, _varint(data_len.size() + enc_pid.size() + buf_.size()));
    append(out, data_len);
    append(out, enc_pid);
    append(out, buf_);
  } else {
    LOG(FATAL) << "Not implemented: Zlib compression (check python client)";
  }
  buf_.clear();
  return out;
}

////////////////
// Packets

vector<uint8_t> Encoder::handshakePacket(const string& host, int port, bool status) {
  varint(PROTO_VERSION);
  str(host);
  uint16(port);
  varint(status ? 1 : 2);  // 1=status, 2=login
  return packet(0x00, 0);
}

vector<uint8_t> Encoder::loginStartPacket(const string& username) {
  str(username);
  return packet(0x00, 0);
}

vector<uint8_t> Encoder::chatMessagePacket(const string& message) {
  str(message);
  return packet(0x02, THRESHOLD);
}

vector<uint8_t> Encoder::playerPositionPacket(double x, double y, double z, bool onGround) {
  float64(x);
  float64(y);
  float64(z);
  boolean(onGround);
  return packet(0x0d, THRESHOLD);
}

vector<uint8_t> Encoder::teleportConfirmPacket(long teleportId) {
  varint(teleportId);
  return packet(0x00, THRESHOLD);
}

vector<uint8_t> Encoder::keepalivePacket(long keepaliveId) {
  uint64(keepaliveId);
  return packet(0x0b, THRESHOLD);
}

vector<uint8_t> Encoder::heldItemChangePacket(uint8_t i) {
  uint16(i);
  return packet(0x1a, THRESHOLD);
}

vector<uint8_t> Encoder::creativeInventoryActionPacket(int16_t index, Slot s) {
  uint16(index);
  slot(s);
  return packet(0x1b, THRESHOLD);
}

vector<uint8_t> Encoder::playerBlockPlacementPacket(BlockPos pos) {
  position(pos);
  varint(1);     // top face
  varint(0);     // main hand
  float32(0.5);  // cursor x
  float32(0.5);  // cursor y
  float32(0.5);  // cursor z
  return packet(0x1f, THRESHOLD);
}

vector<uint8_t> Encoder::playerUseEntityPacket(BlockPos pos) {
  varint(0);       // what is data type target?
  varint(2);       // type: interact
  float32(pos.x);  // target x
  float32(pos.y);  // target y
  float32(pos.z);  // target z
  varint(0);       // main hand
  return packet(0x0a, THRESHOLD);
}

vector<uint8_t> Encoder::playerUseItemPacket() {
  varint(0);  // main hand
  return packet(0x20, THRESHOLD);
}

vector<uint8_t> Encoder::playerLookPacket(float yaw, float pitch, bool onGround) {
  float32(yaw);
  float32(pitch);
  boolean(onGround);
  return packet(0x0f, THRESHOLD);
}

vector<uint8_t> Encoder::playerStartDiggingPacket(BlockPos pos) {
  varint(0);  // start digging
  position(pos);
  byte(1);  // top face
  return packet(0x14, THRESHOLD);
}

vector<uint8_t> Encoder::playerFinishedDiggingPacket(BlockPos pos) {
  varint(2);  // finished digging
  position(pos);
  byte(1);  // top face
  return packet(0x14, THRESHOLD);
}

vector<uint8_t> Encoder::playerDropItemStackInHandPacket() {
  varint(3);          // drop the entire selected stack
  position(0, 0, 0);  // location is always set to 0/0/0
  byte(0);            // face is always set to -Y(0)
  return packet(0x14, THRESHOLD);
}

vector<uint8_t> Encoder::playerDropItemInHandPacket() {
  varint(4);          // drop the selected item
  position(0, 0, 0);  // location is always set to 0/0/0
  byte(0);            // face is always set to -Y(0)
  return packet(0x14, THRESHOLD);
}

vector<uint8_t> Encoder::playerDropItemStackPacket(Slot slot) {
  return creativeInventoryActionPacket(-1, slot);
}

vector<uint8_t> Encoder::playerSetInventorySlotPacket(int16_t index, Slot slot) {
  return creativeInventoryActionPacket(index, slot);
}

vector<uint8_t> Encoder::clickWindowPacket(uint8_t windowId, uint16_t idx, bool rightClick,
                                           uint16_t counter, Slot clicked) {
  byte(windowId);
  uint16(idx);
  byte(rightClick ? 1 : 0);
  uint16(counter);
  varint(0);  // mode
  slot(clicked);
  return packet(0x7, THRESHOLD);
}

vector<uint8_t> Encoder::closeWindowPacket(uint8_t windowId) {
  byte(windowId);
  return packet(0x8, THRESHOLD);
}

////////////////
// Private

vector<uint8_t> Encoder::_varint(long x) {
  vector<uint8_t> out;
  while (true) {
    uint8_t t = x & 127;
    x >>= 7;
    if (x != 0) {
      t |= 128;
    }
    out.push_back(t);
    if (x == 0) {
      return out;
    }
  }
}

void Encoder::bigEndian(unsigned long x, size_t n) {
  for (int i = n - 1; i >= 0; i--) {
    buf_.push_back((x >> (8 * i)) & 0xff);
  }
}
