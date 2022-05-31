#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import a0


if __name__ == "__main__":
    s = a0.SubscriberSync("latest_robot_state", a0.INIT_MOST_RECENT)

    seconds = int(s.read().payload)

    time_diff = time.time() - seconds

    num_seconds_stale = 10
    assert time_diff < num_seconds_stale
