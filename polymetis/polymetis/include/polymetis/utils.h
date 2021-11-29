// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef GRPC_CONTROLLER_MANAGER_UTILS_H
#define GRPC_CONTROLLER_MANAGER_UTILS_H

#include <chrono>
#include <istream>
#include <streambuf>
#include <vector>

#include "polymetis.grpc.pb.h"

/**
Circular buffer class. Preallocates a std::vector with a certain capacity, then
around. Returns NULL* if attempting to an element that is too stale (i.e. when
buffer is above capacity, it drops the earliest elements).
*/
template <typename T> class CircularBuffer {
private:
  std::vector<T> elems;
  ulong index = 0;

public:
  CircularBuffer<T>(int capacity) { elems.reserve(capacity); }

  std::size_t capacity() { return elems.capacity(); }

  void clear() {
    index = 0;
    elems.clear();
  }

  void append(T elem) {
    elems[index % capacity()] = elem;
    index++;
  }

  T *get(int i) {
    if (i >= index || index - i > capacity()) {
      return NULL;
    }
    return &elems[i % capacity()];
  }

  ulong size() { return index; }
};

/*
Time utilities
*/

/**
Returns  nanoseconds as int.
*/
inline long int getNanoseconds() {
  auto epoch_time =
      std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_time)
      .count();
}

/**
Sets timestamp to current time.
*/
inline bool setTimestampToNow(google::protobuf::Timestamp *timestamp_ptr) {
  long int ns = getNanoseconds();
  timestamp_ptr->set_seconds(ns / 1e9);
  timestamp_ptr->set_nanos(ns % (long int)1e9);
  return true;
}

/**
Argument parser, taken from https://stackoverflow.com/a/868894
*/
class InputParser {
public:
  InputParser(int &argc, char **argv) {
    for (int i = 1; i < argc; ++i)
      this->tokens.push_back(std::string(argv[i]));
  }
  /// @author iain
  const std::string &getCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }
  /// @author iain
  bool cmdOptionExists(const std::string &option) const {
    return std::find(this->tokens.begin(), this->tokens.end(), option) !=
           this->tokens.end();
  }

private:
  std::vector<std::string> tokens;
};

#endif