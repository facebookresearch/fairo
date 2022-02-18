// (c) Facebook, Inc. and its affiliates.

#pragma once
#include "./optional.hpp"
#include <chrono>

template <typename Clock = std::chrono::steady_clock> class PeriodicEventT {
public:
  explicit PeriodicEventT(double period) : period_(period) {}

  operator bool() {
    auto now = Clock::now();
    if (!last_event) {
      last_event = now;
    }

    double since_last =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            now - last_event.value())
            .count();
    if (since_last >= period_) {
      last_event = now;
      return true;
    }
    return false;
  }

  // Changes the timer's period.  Does not reset the period timer.
  void setPeriod(double new_period) { period_ = new_period; }

private:
  double period_;
  std::optional<typename Clock::time_point> last_event;
};

using PeriodicEvent = PeriodicEventT<>;
