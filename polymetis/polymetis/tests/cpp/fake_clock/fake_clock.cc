#include "fake_clock.hh"

namespace testing {

  fake_clock::time_point fake_clock::now_us_;
  const bool fake_clock::is_steady = false;

  void fake_clock::advance(duration d) noexcept {
    now_us_ += d;
  }

  void fake_clock::reset_to_epoch() noexcept {
    now_us_ -= (now_us_ - time_point());
  }

  fake_clock::time_point fake_clock::now() noexcept {
    return now_us_;
  }
}
