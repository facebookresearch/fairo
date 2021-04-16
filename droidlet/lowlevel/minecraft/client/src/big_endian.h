// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

class BigEndian {
 public:
  // Read n bytes starting from b in big-endian order
  // N.B. This assumes n <= 8
  static uint64_t readBigEndian(const uint8_t* b, int n) {
    uint64_t out = 0;
    for (int i = 0; i < n; i++) {
      out <<= 8;
      out |= b[i];
    }
    return out;
  }

  // Read a big-endian encoded integer type, e.g.
  //
  // uint16_t x = BigEndian::read<uint16_t>(&buf);
  template <typename T>
  static T readIntType(const uint8_t* b) {
    return readBigEndian(b, sizeof(T));
  }

  static float readFloat(const uint8_t* b) { return readFloatingPoint<float>(b); }
  static double readDouble(const uint8_t* b) { return readFloatingPoint<double>(b); }

 private:
  template <typename T>
  static T readFloatingPoint(const uint8_t* b) {
    uint64_t bytes = readBigEndian(b, sizeof(T));
    T t;
    memcpy(&t, &bytes, sizeof(T));
    return t;
  }
};
