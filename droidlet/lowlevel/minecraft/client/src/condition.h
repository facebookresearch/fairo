// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <condition_variable>
#include <mutex>

class Condition {
 public:
  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    while (!condition_) {
      cv_.wait(lock);
    }
  }

  void trigger() {
    condition_ = true;
    cv_.notify_all();
  }

  void clear() { condition_ = false; }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  bool condition_ = false;
};
