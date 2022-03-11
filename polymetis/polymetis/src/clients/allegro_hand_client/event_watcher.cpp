// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "event_watcher.hpp"

void EventWatcher::observe(std::string event_name) {
  event_data_[event_name].observe();
}

void EventWatcher::checkpoint() {
  for (auto &entry : event_data_) {
    entry.second.checkpoint();
  }
}

std::ostream &operator<<(std::ostream &os, const EventWatcher &watcher) {
  for (auto &entry : watcher.event_data_) {
    os << entry.first << ": " << entry.second << std::endl;
  }
  return os;
}

EventWatcher gEventWatcher;
