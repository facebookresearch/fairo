// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <spdlog/spdlog.h>

#define VALIDATE(x)                                                            \
  (!(x) && (spdlog::error("VALIDATE(" #x ") failed."), exit(1), true))
#define VALIDATE_EQ(x, y)                                                      \
  (((x) != (y)) &&                                                             \
   (spdlog::error("VALIDATE_EQ(" #x ", " #y ") failed ({} != {}).",            \
                  std::to_string((x)), std::to_string((y))),                   \
    exit(1), true))

template <class ArgType>
inline ArgType median3(ArgType a, ArgType b, ArgType c) {
  if (a >= b) {   // b - a
    if (b >= c) { // c - b - a
      return b;
    } else if (c >= a) { // b - a - c
      return a;
    }
  }

  if (b >= a) {   // a - b
    if (a >= c) { // c - a - b
      return a;
    } else if (c >= b) { // a - b - c
      return b;
    }
  }

  return c;
}
