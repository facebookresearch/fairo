// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "allegro_hand.hpp"
#include "digital_filter.hpp"

class FingerVelocityFilter {
public:
  FingerVelocityFilter(YAML::Node config) {
    DigitalFilter butter = loadFilter(config).value();
    butter.resize(kNJoint);
    for (int i = 0; i < kNFinger; i++) {
      butter_.push_back(butter);
      diff_.push_back(DifferenceFilter(kNJoint));
    }

    dt_ = config["dt"].as<double>();
  }

  void filter_position(const Eigen::VectorXd &position, int finger,
                       Eigen::Ref<Eigen::VectorXd> velocity) {
    if (!initialized_[finger]) {
      butter_[finger].reset(position.segment(finger * kNJoint, kNJoint));
      diff_[finger].reset(position.segment(finger * kNJoint, kNJoint));
      initialized_[finger] = true;
    }

    butter_[finger].filter(position.segment(finger * kNJoint, kNJoint),
                           &smooth_pos_);
    diff_[finger].filter(smooth_pos_, &smooth_vel_);
    velocity.segment(finger * kNJoint, kNJoint) = smooth_vel_ / dt_;
  }

private:
  std::array<bool, kNFinger> initialized_;
  std::vector<DigitalFilter> butter_;
  std::vector<DifferenceFilter> diff_;
  Eigen::VectorXd smooth_pos_{kNJoint};
  Eigen::VectorXd smooth_vel_{kNJoint};
  double dt_;
};
