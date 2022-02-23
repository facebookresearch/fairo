// A clock class that is convenient for testing.
//
// This class satisfies the TrivialClock requirement
// (http://en.cppreference.com/w/cpp/concept/TrivialClock) and as such
// can be used in place of any standard clock
// (e.g. std::chrono::system_clock).
//
// The clock uses an uint64_t internally, so it can store all
// nanoseconds in a century. This is consistent with the precision
// required of std::chrono::nanoseconds in C++11.
//
// Example usage:
//
//    fake_clock::time_point t1 = fake_clock::now();
//    fake_clock::advance(std::chrono::milliseconds(100));
//    fake_clock::time_point t2 = fake_clock::now();
//    auto elapsed_us = std::chrono::duration_cast<
//           std::chrono::microseconds>(t2 - t1).count();
//    assert(100000 == elapsed_us);

#ifndef FAKE_CLOCK_HH__
#define FAKE_CLOCK_HH__

#include <chrono>
#include <cstdint>

namespace testing {
  class fake_clock {
   public:
    typedef uint64_t rep;
    typedef std::ratio<1l, 1000000000l> period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<fake_clock> time_point;

    static void advance(duration d) noexcept;
    static void reset_to_epoch() noexcept;
    static time_point now() noexcept;

   private:
    fake_clock() = delete;
    ~fake_clock() = delete;
    fake_clock(fake_clock const&) = delete;

    static time_point now_us_;
    static const bool is_steady;
  };
}

#endif
