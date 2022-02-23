#include "fake_clock.hh"
#include <climits>
#include <gtest/gtest.h>

namespace {

  using testing::fake_clock;

  class FakeClockTest : public ::testing::Test {
    void SetUp() {
      // For tests that use the clock, it may be advisable to use
      // this. The tests in this file don't actually need it.
      fake_clock::reset_to_epoch();
    }
  };

  TEST_F(FakeClockTest, ClockTest) {
    fake_clock::time_point t0 = fake_clock::now();
    fake_clock::time_point t1 = fake_clock::now();
    EXPECT_EQ(std::chrono::microseconds(0), t1 - t0);

    fake_clock::advance(std::chrono::microseconds(10));
    fake_clock::time_point t2 = fake_clock::now();
    EXPECT_EQ(std::chrono::microseconds(10), t2 - t0);

    fake_clock::reset_to_epoch();
    fake_clock::time_point t3 = fake_clock::now();
    EXPECT_EQ(std::chrono::microseconds(0), t3 - t0);
  }

  TEST_F(FakeClockTest, ExampleTest) {
    // Ensures that the example in README.md works.
    fake_clock::time_point t1 = fake_clock::now();
    fake_clock::advance(std::chrono::milliseconds(100));
    fake_clock::time_point t2 = fake_clock::now();
    int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    EXPECT_EQ(100000, elapsed_us);
  }

  TEST_F(FakeClockTest, PrecisionTest) {
    // Ensures that there are enough bits in the internal
    // representation of time to store all nanoseconds in a century.
    EXPECT_TRUE(sizeof(fake_clock::rep) * CHAR_BIT >= 62);
  }
}
