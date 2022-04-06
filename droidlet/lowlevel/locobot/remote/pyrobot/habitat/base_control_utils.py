# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class LocalActionStatus:
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    FREE = 5
    UNKOWN = 6
    PREEMPTING = 7
    DISABLED = 8


import threading
import copy


class LocalActionServer(object):
    """docstring for LocalActionServer"""

    def __init__(self):

        self._lock = threading.RLock()
        self._state = LocalActionStatus.UNKOWN

    def is_preempt_requested(self):
        if (
            self.get_state() == LocalActionStatus.PREEMPTING
            or self.get_state() == LocalActionStatus.DISABLED
        ):
            return True
        else:
            return False

    def get_state(self):
        self._lock.acquire()
        state = copy.copy(self._state)
        self._lock.release()
        return state

    def _set_state(self, state):
        self._lock.acquire()
        self._state = state
        self._lock.release()

    def cancel_goal(self):
        self._set_state(LocalActionStatus.PREEMPTING)

    def set_preempted(self):
        self._set_state(LocalActionStatus.PREEMPTED)

    def set_succeeded(self):
        self._set_state(LocalActionStatus.SUCCEEDED)

    def set_aborted(self):
        self._set_state(LocalActionStatus.ABORTED)

    def set_active(self):
        self._set_state(LocalActionStatus.ACTIVE)

    def is_disabled(self):
        state = self.get_state()
        if state == LocalActionStatus.DISABLED:
            return True
        return False

    def disable(self):
        self._set_state(LocalActionStatus.DISABLED)

    def is_available(self):
        state = self.get_state()
        if (
            state == LocalActionStatus.ACTIVE
            or state == LocalActionStatus.PREEMPTING
            or state == LocalActionStatus.DISABLED
        ):
            return False
        else:
            return True
