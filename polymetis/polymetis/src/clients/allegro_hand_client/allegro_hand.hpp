// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "pcan_netdev_interface.hpp"
#include "periodic_event.hpp"

constexpr int kNFinger = 4;
constexpr int kNJoint = 4;
constexpr int kNDofs = kNFinger * kNJoint;

// Interface to Allegro Hand 4.0
// Joint and Torque state vectors are indexed as:
// [finger_idx * kNJoint + joint_idx]
// Fingers are indexed in order:
// [index, middle, ring, thumb] (there is no 5th finger)
class AllegroHand {
public:
  // n.b. The firmware appears to only support periodic broadcast of joint
  // position
  virtual void setPeriodicRead(uint16_t millis) = 0;

  // n.b. The 'servo enable' command doesn't seem to have any effect
  virtual void setServoEnable(bool enable) = 0;

  // Sets the torques that will be sent to the hand.
  virtual void setTorques(double *torques) = 0;
  virtual void requestStatus() = 0;

  // Polls for new state from the hand. Sends joint torques
  // to each finger if new state arrived from that finger on
  // the previous poll.
  // Returns -1 or index of finger from which data was received.
  virtual int poll() = 0;

  // Sends torques for joints on the given finger
  virtual void sendTorque(int finger) = 0;

  // Gets a pointer to an array of joint positions.
  virtual double *getPositions() = 0;

  // Returns true if all the fingers have received state updates since last
  // reset
  virtual bool allStatesUpdated() const = 0;

  // Reset the state update tracker
  virtual void resetStateUpdateTracker() = 0;

  // Reset the communication channel
  virtual void resetCommunication(){};
};

class AllegroHandImpl : public AllegroHand {
public:
  explicit AllegroHandImpl(PcanInterface &&pcan);
  // n.b. The firmware appears to only support periodic broadcast of joint
  // position
  void setPeriodicRead(uint16_t millis) final;

  // n.b. The 'servo enable' command doesn't seem to have any effect
  void setServoEnable(bool enable) final;

  // Sets the torques that will be sent to the hand.
  void setTorques(double *torques) final;
  void requestStatus() final;

  // Polls for new state from the hand. Sends joint torques
  // to each finger if new state arrived from that finger on
  // the previous poll.
  // Returns -1 or index of finger from which data was received.
  int poll() final;

  // Sends torques for joints on the given finger
  void sendTorque(int finger) final;

  // Gets a pointer to an array of joint positions.
  double *getPositions() final;

  // Returns true if all the fingers have received state updates since last
  // reset
  bool allStatesUpdated() const final;

  // Reset the state update tracker
  void resetStateUpdateTracker() final {
    memset(state_updated_, 0, sizeof(state_updated_));
  }

  void resetCommunication() final;

private:
  bool send(unsigned char msg_id, char data_len = 0, const void *data = NULL,
            bool expect_return = false);
  void initialize();
  inline double convertPosition(int16_t raw_pos) const;

  double torques_[kNDofs];
  double position_[kNDofs];
  bool send_pending_[kNFinger];
  const unsigned char device_id_ = 0;
  bool state_updated_[kNFinger];

  PcanInterface pcan_;
};

class MockAllegroHand : public AllegroHand {
public:
  void setPeriodicRead(uint16_t millis) override {
    timer_.setPeriod(millis * 1000.0);
  }

  void setServoEnable(bool enable) override {}
  void setTorques(double *torques) override {}
  void requestStatus() override {}
  int poll() override {
    if (timer_) {
      next_finger_ = (next_finger_ + 1) % kNFinger;
      if (next_finger_ == kNFinger - 1) {
        all_updated_ = true;
      }
      return next_finger_;
    }
    return -1;
  }

  void sendTorque(int finger) override {}

  double *getPositions() override { return positions_; }

  bool allStatesUpdated() const override { return all_updated_; }

  void resetStateUpdateTracker() override { all_updated_ = false; }

private:
  PeriodicEvent timer_{1 / 1200.0};
  int next_finger_{0};
  bool all_updated_{false};
  double positions_[kNDofs];
};
