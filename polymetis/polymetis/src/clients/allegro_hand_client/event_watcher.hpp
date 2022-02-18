// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <ostream>
#include <string>
#include <unordered_map>

template <typename Clock = std::chrono::steady_clock> class EventDataT {
public:
  EventDataT() { checkpoint(); }

  void observe() {
    count_++;
    count_since_last_checkpoint_++;
  }

  void checkpoint() {
    last_checkpoint_time_ = Clock::now();
    count_since_last_checkpoint_ = 0;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const EventDataT<Clock> &data) {
    double time_elapsed = data.secondsSinceCheckpoint();
    os << "Count: " << data.count_since_last_checkpoint_;
    os << " Rate: ";
    if (time_elapsed > 0) {
      os << data.count_since_last_checkpoint_ / time_elapsed;
    } else {
      os << "<no time elapsed>";
    }
    return os;
  }

  inline int64_t countSinceCheckpoint() const {
    return count_since_last_checkpoint_;
  }

  inline double secondsSinceCheckpoint() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               Clock::now() - last_checkpoint_time_)
        .count();
  }

  inline double rateSinceCheckpoint() const {
    if (secondsSinceCheckpoint() > 0) {
      return count_since_last_checkpoint_ / secondsSinceCheckpoint();
    }
    return 0;
  }

  inline int64_t count() const { return count_; }

private:
  typename Clock::time_point last_checkpoint_time_;
  int64_t count_{0};
  int64_t count_since_last_checkpoint_{0};
};

using EventData = EventDataT<>;

class EventWatcher {
public:
  void observe(std::string event_name);
  void checkpoint();
  friend std::ostream &operator<<(std::ostream &os,
                                  const EventWatcher &watcher);

private:
  std::unordered_map<std::string, EventData> event_data_;
};

extern EventWatcher gEventWatcher;
