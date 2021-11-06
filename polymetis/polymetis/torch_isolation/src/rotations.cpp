// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rotations.hpp"

extern "C" {

#define EPSILON 1e-36

// Tensor & Eigen utils
Eigen::Quaternionf tensor4ToQuat(torch::Tensor T) {
  assert(T.dim() == 1);
  assert(T.size(0) == 4);
  return Eigen::Quaternionf(
      Eigen::Vector4f(Eigen::Map<Eigen::Matrix<float, 1, 4, Eigen::RowMajor>>(
          T.to(torch::kFloat32).data_ptr<float>(), 1, 4)));
}

Eigen::Matrix3f tensor33ToMatrix(torch::Tensor T) {
  assert(T.dim() == 2);
  assert(T.size(0) == 3);
  assert(T.size(1) == 3);
  return Eigen::Matrix3f(
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
          T.to(torch::kFloat32).data_ptr<float>(), 3, 3));
}

Eigen::AngleAxisf tensor3ToAngleAxis(torch::Tensor T) {
  assert(T.dim() == 1);
  assert(T.size(0) == 3);
  auto r =
      Eigen::Vector3f(Eigen::Map<Eigen::Matrix<float, 1, 3, Eigen::RowMajor>>(
          T.to(torch::kFloat32).data_ptr<float>(), 1, 3));
  auto r_norm = r.norm() + EPSILON;
  return Eigen::AngleAxisf(r_norm, r / r_norm);
}

Eigen::Vector4f quatToVector(Eigen::Quaternionf q) {
  Eigen::Vector4f v;
  v(0) = q.x();
  v(1) = q.y();
  v(2) = q.z();
  v(3) = q.w();
  return v;
}

// API functions

torch::Tensor normalizeQuaternion(torch::Tensor q) {
  return torch::from_blob(quatToVector(tensor4ToQuat(q).normalized()).data(),
                          {4}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor invertQuaternion(torch::Tensor q) {
  auto result = quatToVector(tensor4ToQuat(q).inverse());
  return torch::from_blob(result.data(), {4}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor quatToAxis(torch::Tensor q) {
  Eigen::AngleAxisf aa(tensor4ToQuat(q));
  Eigen::Vector3f result = aa.axis();
  return torch::from_blob(result.data(), {3}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor quatToAngle(torch::Tensor q) {
  Eigen::AngleAxisf aa(tensor4ToQuat(q));
  return torch::from_blob(&aa.angle(), {1}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor quatToMatrix(torch::Tensor q) {
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> result =
      tensor4ToQuat(q).toRotationMatrix();
  return torch::from_blob(result.data(), {3, 3}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor quatToRotvec(torch::Tensor q) {
  Eigen::AngleAxisf r(tensor4ToQuat(q));
  Eigen::Vector3f result = r.angle() * r.axis();
  return torch::from_blob(result.data(), {3}, torch::kFloat32)
      .to(q.dtype())
      .clone();
}

torch::Tensor matrixToQuat(torch::Tensor m) {
  Eigen::Quaternionf q(tensor33ToMatrix(m));
  auto result = quatToVector(q.normalized());
  return torch::from_blob(result.data(), {4}, torch::kFloat32)
      .to(m.dtype())
      .clone();
}

torch::Tensor rotvecToQuat(torch::Tensor r) {
  Eigen::Quaternionf q(tensor3ToAngleAxis(r));
  auto result = quatToVector(q.normalized());
  return torch::from_blob(result.data(), {4}, torch::kFloat32)
      .to(r.dtype())
      .clone();
}

torch::Tensor quaternionMultiply(torch::Tensor q1, torch::Tensor q2) {
  auto result =
      quatToVector(tensor4ToQuat(q1) * tensor4ToQuat(q2).normalized());
  return torch::from_blob(result.data(), {4}, torch::kFloat32)
      .to(q1.dtype())
      .clone();
}

// Binding

TORCH_LIBRARY(torchrot, m) {
  m.def("normalize_quaternion", normalizeQuaternion)
      .def("invert_quaternion", invertQuaternion)
      .def("quat2axis", quatToAxis)
      .def("quat2angle", quatToAngle)
      .def("quat2matrix", quatToMatrix)
      .def("quat2rotvec", quatToRotvec)
      .def("matrix2quat", matrixToQuat)
      .def("rotvec2quat", rotvecToQuat)
      .def("quaternion_multiply", quaternionMultiply);
}

} /* extern "C" */
