// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <cstdio>

#include "digital_filter.hpp"
#include "event_watcher.hpp"
#include "pcan_netdev_interface.hpp"
#include "periodic_event.hpp"

#include "./fake_clock/fake_clock.hh"

TEST(PeriodicEvent, basic) {
  PeriodicEventT<testing::fake_clock> event(25);

  EXPECT_FALSE((bool)event);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 25; j++) {
      EXPECT_FALSE((bool)event);
      testing::fake_clock::advance(std::chrono::seconds(1));
    }
    EXPECT_TRUE((bool)event);
  }
}

TEST(EventData, basic) {
  EventDataT<testing::fake_clock> event_data;

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.count(), 0);

  testing::fake_clock::advance(std::chrono::seconds(2));

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 2);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 0);

  event_data.observe();
  event_data.observe();
  event_data.observe();

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 2);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 3 / 2.0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 3);

  testing::fake_clock::advance(std::chrono::seconds(2));

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 4);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 3 / 4.0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 3);

  event_data.checkpoint();

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.count(), 3);

  event_data.observe();

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 1);

  testing::fake_clock::advance(std::chrono::seconds(5));

  EXPECT_EQ(event_data.secondsSinceCheckpoint(), 5);
  EXPECT_EQ(event_data.rateSinceCheckpoint(), 1 / 5.0);
  EXPECT_EQ(event_data.countSinceCheckpoint(), 1);
  EXPECT_EQ(event_data.count(), 4);
}

void TestPcan(std::string device_name) {
  try {
    testing::internal::CaptureStdout();
    PcanInterface pcan(device_name);

    EXPECT_FALSE(pcan.readPcan(nullptr));
    EXPECT_NE(testing::internal::GetCapturedStdout().find(
                  "CAN Bus Error: A parameter contains an invalid value"),
              std::string::npos);

    testing::internal::CaptureStdout();
    can_frame msg;
    if (!pcan.readPcan(&msg)) {
      const std::string &capturedStdout =
          testing::internal::GetCapturedStdout();
      if (capturedStdout.length() != 0) {
        EXPECT_NE(capturedStdout.find(
                      "CAN Bus Error: A PCAN Channel has not been initialized "
                      "yet or the initialization process has failed"),
                  std::string::npos);
      }
    } else {
      testing::internal::GetCapturedStdout();
    }
    testing::internal::CaptureStdout();
    msg.can_id = 0;
    msg.can_dlc = 1;
    if (!pcan.writePcan(msg)) {
      EXPECT_NE(testing::internal::GetCapturedStdout().find(
                    "CAN Bus Error: A PCAN Channel has not been initialized "
                    "yet or the initialization process has failed"),
                std::string::npos);
    } else {
      testing::internal::GetCapturedStdout();
    }

    testing::internal::CaptureStdout();
    msg.can_id = 0;
    msg.can_dlc = 10;
    if (!pcan.writePcan(msg)) {
      EXPECT_NE(testing::internal::GetCapturedStdout().find("Bad MSG Len"),
                std::string::npos);
    } else {
      testing::internal::GetCapturedStdout();
    }
  } catch (const std::runtime_error &e) {
    testing::internal::GetCapturedStdout();
    EXPECT_STREQ(e.what(), "Failed to initialize CAN interface");
  }
}

TEST(PCAN, BasicFailures) { TestPcan("bad_can_id"); }

TEST(DigitalFilter, Butter) {
  Eigen::VectorXd a(3);
  Eigen::VectorXd b(3);

  a << 1.000000000000000, -1.561018075800718, 0.641351538057563;
  b << 0.0200833655642112, 0.0401667311284225, 0.0200833655642112;

  DigitalFilter filt(a, b, 10);

  Eigen::VectorXd obs(10);
  Eigen::VectorXd out(10);
  for (int i = 0; i < 10; i++) {
    obs(i) = i;
  }

  filt.reset(obs);
  filt.filter(obs, &out);
  for (int i = 0; i < 10; i++) {
    ASSERT_NEAR(obs(i), out(i), 1e-5);
  }
  for (int i = 0; i < 10; i++) {
    filt.filter(obs, &out);
  }
  for (int i = 0; i < 10; i++) {
    ASSERT_NEAR(obs(i), out(i), 1e-5);
  }
  obs.setZero();
  filt.filter(obs, &out);

  for (int i = 0; i < 10; i++) {
    ASSERT_NEAR(out(i), i * 0.979916634435798, 1e-5);
  }

  filt.filter(obs, &out);

  for (int i = 0; i < 10; i++) {
    ASSERT_NEAR(out(i), i * 0.908399406638728, 1e-5);
  }
}

TEST(DigitalFilter, Difference) {
  DifferenceFilter filt(2);
  Eigen::Vector2d obs;
  Eigen::VectorXd diff(2);
  obs << 1, 2;
  filt.reset(obs);
  filt.filter(obs, &diff);
  ASSERT_EQ(diff(0), 0);
  ASSERT_EQ(diff(1), 0);

  obs << 10, 20;
  filt.filter(obs, &diff);
  ASSERT_EQ(diff(0), 9);
  ASSERT_EQ(diff(1), 18);
}

TEST(DigitalFilter, SlewLimiter) {
  SlewLimiter limiter(.1, 2);
  Eigen::Vector2d obs;
  Eigen::Vector2d out;

  obs << 1, 2;
  out << 0, 3;
  limiter.reset(out);
  limiter.filter(obs, out);
  ASSERT_EQ(out(0), .1);
  ASSERT_EQ(out(1), 2.9);

  for (int i = 0; i < 500; i++) {
    limiter.filter(obs, out);
  }

  ASSERT_EQ(out(0), obs(0));
  ASSERT_EQ(out(1), obs(1));
}

TEST(MathUtil, Median3) {
  EXPECT_EQ(median3(1, 2, 3), 2);
  EXPECT_EQ(median3(1, 3, 2), 2);
  EXPECT_EQ(median3(2, 1, 3), 2);
  EXPECT_EQ(median3(2, 3, 1), 2);
  EXPECT_EQ(median3(3, 1, 2), 2);
  EXPECT_EQ(median3(3, 2, 1), 2);
  EXPECT_EQ(median3(0, 0, 2), 0);
}

TEST(MedianFilter, MedianFilterTest) {
  MedianFilter filter;
  Eigen::VectorXd obs(5);
  Eigen::VectorXd out(obs.size());
  filter.resize(obs.size());
  for (int i = 0; i < 10; i++) {
    for (int c = 0; c < obs.size(); c++) {
      obs(c) = i + c;
    }
    filter.filter(obs, &out);
    for (int c = 0; c < obs.size(); c++) {
      EXPECT_EQ(out(c), i == 0 ? 0 : (i - 1) + c);
    }
  }
}
