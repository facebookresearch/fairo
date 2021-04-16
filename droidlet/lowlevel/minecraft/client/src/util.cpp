// Copyright (c) Facebook, Inc. and its affiliates.

#include "util.h"
#include <math.h>
#include <utility>

// This is equal to python's divmod(x, y), but c++'s div/mod work
// differently on negative numbers.
// Here, mod will always be positive
std::pair<int, int> pyDivmod(int a, int b) {
  if (a < 0) {
    int div = ((a + 1) / b) - 1;
    int mod = ((a + 1) % b) + (b - 1);
    return std::make_pair(div, mod);
  } else {
    return std::make_pair(a / b, a % b);
  }
}

BlockPos discreteStepDirection(float yaw) {
  int x = lround(-sin(yaw * M_PI / 180));
  int z = lround(cos(yaw * M_PI / 180));
  return {x, 0, z};
}
