// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

template <typename T>
class BlockingQueue {
 public:
  void push(const T& t) {
    std::unique_lock<std::mutex> lock(m_);
    q_.push(t);
    lock.unlock();
    c_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(m_);
    while (q_.empty()) {
      c_.wait(lock);
    }
    T t = q_.front();
    q_.pop();
    lock.unlock();
  }

  std::vector<T> popall() {
    std::unique_lock<std::mutex> lock(m_);
    size_t n = q_.size();
    std::vector<T> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
      v.push_back(q_.front());
      q_.pop();
    }
    lock.unlock();
    return v;
  }

 private:
  std::condition_variable c_;
  std::mutex m_;
  std::queue<T> q_;
};
