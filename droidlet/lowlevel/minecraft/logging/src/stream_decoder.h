// Copyright (c) Facebook, Inc. and its affiliates.


#pragma once
#include <fstream>
#include <istream>
#include <string>
#include <vector>

#include "../../client/src/big_endian.h"

class StreamDecoder {
  static const long BUF_SIZE = 4096;

 public:
  StreamDecoder(const std::string& path) : is_(path, std::ifstream::binary) {}

  void skip(int n) { read(n); }

  // Peek n bytes, writing to dest. Return true if valid, false if EOF.
  bool peek(std::vector<uint8_t>& dest, int n) {
    auto start = is_.tellg();
    bool valid = read(n);
    if (!valid) return false;
    dest.reserve(n);
    memcpy(&dest[0], buf_, n);
    is_.seekg(start);
    return true;
  }

  unsigned long getCount() { return is_.tellg(); }

  // Primitives

  template <typename T>
  T readIntType() {
    read(sizeof(T));
    return BigEndian::readIntType<T>(buf_);
  }

  float readFloat() {
    read(sizeof(float));
    return BigEndian::readFloat(buf_);
  }

  double readDouble() {
    read(sizeof(double));
    return BigEndian::readDouble(buf_);
  }

  // Logging Compound Types

  std::string readString() {
    auto length = readIntType<uint16_t>();
    read(length + 1);
    std::string s((char*)buf_, length);
    CHECK_EQ(buf_[length], 0) << "String not null-terminated: " << s;
    return s;
  }

  BlockPos readIntPos() {
    return {static_cast<int>(readIntType<int64_t>()), static_cast<int>(readIntType<int64_t>()),
            static_cast<int>(readIntType<int64_t>())};
  }

  Pos readFloatPos() { return {readDouble(), readDouble(), readDouble()}; }

  Look readLook() { return {readFloat(), readFloat()}; }

  Block readBlock() { return {readIntType<uint8_t>(), readIntType<uint8_t>()}; }

  Slot readItem() {
    auto id = readIntType<uint16_t>();
    auto count = readIntType<uint16_t>();
    auto meta = readIntType<uint16_t>();  // supposed to be damage, but seems to be meta
    return {id, static_cast<uint8_t>(meta), static_cast<uint8_t>(count), 0};
  }

 private:
  // Return true if success, false if EOF
  bool read(int n) {
    CHECK_LT(n, BUF_SIZE);
    is_.read((char*)buf_, n);
    if (is_.gcount() == 0) {
      return false;
    }
    CHECK_EQ(n, is_.gcount()) << "Unexpected EOF";
    return true;
  }

  std::ifstream is_;
  uint8_t buf_[BUF_SIZE];
};
