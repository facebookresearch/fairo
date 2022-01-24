# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import polymetis_pb2


def check_episode_log(episode_log, timesteps, check_timestamps=False):
    assert len(episode_log) == (
        timesteps
    ), f"episode length={len(episode_log)}, but expected={timesteps}"
    assert type(episode_log) is list
    assert type(episode_log[0]) is polymetis_pb2.RobotState

    if check_timestamps:
        for i in range(len(episode_log) - 1):
            timestamp_prev = episode_log[i].timestamp.ToDatetime()
            timestamp_curr = episode_log[i + 1].timestamp.ToDatetime()
            assert (timestamp_curr - timestamp_prev).microseconds > 0
