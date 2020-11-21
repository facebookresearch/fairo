// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <utility>
#include "types.h"

// This is equal to python's divmod(x, y), but c++'s div/mod work
// differently on negative numbers.
// Here, mod will always be positive
std::pair<int, int> pyDivmod(int a, int b);

BlockPos discreteStepDirection(float yaw);
