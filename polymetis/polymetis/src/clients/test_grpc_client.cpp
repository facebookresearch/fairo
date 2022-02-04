// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "spdlog/spdlog.h"
#include <chrono>
#include <numeric>
#include <thread>

#include <malloc.h>
#include <pthread.h>
#include <real_time_tools/thread.hpp>
#include <sched.h>
#include <sys/resource.h>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

using namespace std::chrono_literals;

class TestGrpcClient {
public:
  TestGrpcClient(int num_dofs, std::shared_ptr<grpc::ChannelInterface> channel)
      : stub_(PolymetisControllerServer::NewStub(channel)) {
    int log_iters = 3000;
    std::vector<float> times_taken;
    times_taken.reserve(log_iters);

    float global_max = -99999.0;
    float global_min = 99999.0;
    float global_avg = 0.0;

    int i = 0;

    while (true) {
      auto start = std::chrono::high_resolution_clock::now();

      assert(SendCommand(num_dofs));

      auto end = std::chrono::high_resolution_clock::now();

      float ms_taken =
          ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                       start)
               .count()) /
          1000000.0;
      if (i % log_iters >= times_taken.size()) {
        times_taken.push_back(ms_taken);
      } else {
        times_taken.at(i % log_iters) = ms_taken;
      }
      if (ms_taken > 1.0) {
        spdlog::warn("==== Warning: round trip time takes {} ms! ====",
                     ms_taken);
      }

      if (i > 0 && i % log_iters == 0) {
        float max_time_taken =
            *std::max_element(times_taken.begin(), times_taken.end());
        float min_time_taken =
            *std::min_element(times_taken.begin(), times_taken.end());

        float sum = 0.0;
        for (int j = 0; j < log_iters; j++) {
          sum += times_taken[j];
        }
        float avg_time_taken = sum / log_iters;

        spdlog::info("max: {} ms, min: {} ms, avg: {} ms", max_time_taken,
                     min_time_taken, avg_time_taken);

        if (max_time_taken > global_max) {
          global_max = max_time_taken;
        }
        if (min_time_taken < global_min) {
          global_min = min_time_taken;
        }
        int num_logs_so_far = i / log_iters;
        global_avg = ((num_logs_so_far - 1) * global_avg + avg_time_taken) /
                     (float)num_logs_so_far;
        spdlog::info("global max: {}, min: {}, avg: {}", global_max, global_min,
                     global_avg);
      }
      i++;

      std::this_thread::sleep_until(start + 1ms);
    }
  }

private:
  std::unique_ptr<PolymetisControllerServer::Stub> stub_;
  bool SendCommand(int num_dofs) {
    RobotState state_;
    for (int i = 0; i < num_dofs; i++) {
      state_.add_joint_positions(0.0);
      state_.add_joint_velocities(0.0);
    }

    setTimestampToNow(state_.mutable_timestamp());

    TorqueCommand torque_command;
    ClientContext context;
    Status status = stub_->ControlUpdate(&context, state_, &torque_command);
    if (!status.ok()) {
      spdlog::error("SendCommand failed: {}", status.error_code());
      return false;
    }
    return true;
  }
};

void *RunClient(void *) {
  int num_dofs = 7;
  TestGrpcClient client(
      num_dofs, grpc::CreateChannel("localhost:50051",
                                    grpc::InsecureChannelCredentials()));

  return NULL;
}

int main(int argc, char *argv[]) {
  bool use_real_time = true;
  if (!use_real_time) {
    RunClient(NULL);
  } else {
    real_time_tools::RealTimeThread rt_thread;
    rt_thread.parameters_.stack_size_ = 1024 * 1024 * 200;
    spdlog::info("Using 200MB as stack size");

    rt_thread.parameters_.priority_ = 81;
    spdlog::info("Using RT priority {}", rt_thread.parameters_.priority_);

    if (!mallopt(M_TRIM_THRESHOLD, -1)) {
      return 1;
    } // disable sbrk
    if (!mallopt(M_MMAP_MAX, 0)) {
      return 1;
    } // disable mmap

    rt_thread.create_realtime_thread(RunClient);
    rt_thread.join();
  }

  return 0;
}
