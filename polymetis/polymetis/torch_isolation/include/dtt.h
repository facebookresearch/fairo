/**
 * === Modified from https://github.com/andrewssobral/dtt ===
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * === Original comments & license: ===
 *
 * DTT - Data Transfer Tools for C++ Linear Algebra Libraries.
 * It supports data transfer between the following libraries:
 * Eigen, Armadillo, OpenCV, ArrayFire, LibTorch
 *
 * MIT License
 *
 * Copyright (c) 2019 Andrews Sobral
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#include <Eigen/Dense>
#include <torch/torch.h>

namespace dtt {

// same as MatrixXf, but with row-major memory layout
// typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
// MatrixXf_rm;

// MatrixXrm<float> x; instead of MatrixXf_rm x;
template <typename V>
using MatrixXrm =
    typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// MatrixX<float> x; instead of Eigen::MatrixXf x;
template <typename V>
using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

//---------------------------------------------------------------------------
// Eigen to LibTorch
//---------------------------------------------------------------------------

template <typename V> torch::Tensor eigen2libtorch(MatrixX<V> &M) {
  Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
  std::vector<int64_t> dims = {E.rows(), E.cols()};
  auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
  return T;
}

template <typename V>
torch::Tensor eigen2libtorch(MatrixXrm<V> &E, bool copydata = true) {
  std::vector<int64_t> dims = {E.rows(), E.cols()};
  auto T = torch::from_blob(E.data(), dims);
  if (copydata)
    return T.clone();
  else
    return T;
}

//---------------------------------------------------------------------------
// LibTorch to Eigen
//---------------------------------------------------------------------------

template <typename V>
Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>
libtorch2eigen(torch::Tensor &Tin) {
  /*
   LibTorch is Row-major order and Eigen is Column-major order.
   MatrixXrm uses Eigen::RowMajor for compatibility.
   */
  auto T = Tin.to(torch::kCPU);
  Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
  return E;
}

} // namespace dtt
