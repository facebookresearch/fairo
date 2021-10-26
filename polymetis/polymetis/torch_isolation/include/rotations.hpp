// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef TORCHROT_H
#define TORCHROT_H

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>

#include "dtt.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <torch/script.h>
#include <torch/torch.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

Eigen::Quaternionf tensor4ToQuat(torch::Tensor T);
Eigen::Matrix3f tensor33ToMatrix(torch::Tensor T);
Eigen::AngleAxisf tensor3ToAngleAxis(torch::Tensor T);
Eigen::Vector4f quatToVector(Eigen::Quaternionf q);

torch::Tensor normalizeQuaternion(torch::Tensor q);
torch::Tensor invertQuaternion(torch::Tensor q);
torch::Tensor quatToAxis(torch::Tensor q);
torch::Tensor quatToAngle(torch::Tensor q);
torch::Tensor quatToMatrix(torch::Tensor q);
torch::Tensor quatToRotvec(torch::Tensor q);
torch::Tensor matrixToQuat(torch::Tensor m);
torch::Tensor rotvecToQuat(torch::Tensor r);
torch::Tensor quaternionMultiply(torch::Tensor q1, torch::Tensor q2);

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
