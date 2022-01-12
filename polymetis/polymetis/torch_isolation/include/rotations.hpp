// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#ifndef TORCHROT_H
#define TORCHROT_H

#include <assert.h>
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "dtt.h"
#include <torch/torch.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define C_TORCH_EXPORT __attribute__((visibility("default")))

C_TORCH_EXPORT Eigen::Quaternionf tensor4ToQuat(torch::Tensor T);
C_TORCH_EXPORT Eigen::Matrix3f tensor33ToMatrix(torch::Tensor T);
C_TORCH_EXPORT Eigen::AngleAxisf tensor3ToAngleAxis(torch::Tensor T);
C_TORCH_EXPORT Eigen::Vector4f quatToVector(Eigen::Quaternionf q);

C_TORCH_EXPORT torch::Tensor normalizeQuaternion(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor invertQuaternion(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor quatToAxis(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor quatToAngle(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor quatToMatrix(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor quatToRotvec(torch::Tensor q);
C_TORCH_EXPORT torch::Tensor matrixToQuat(torch::Tensor m);
C_TORCH_EXPORT torch::Tensor rotvecToQuat(torch::Tensor r);
C_TORCH_EXPORT torch::Tensor quaternionMultiply(torch::Tensor q1,
                                                torch::Tensor q2);

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
