// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

#include "Eigen/Eigen"
#include "yaml-cpp/yaml.h"

#include "./optional.hpp"
#include "./utility.hpp"
#include "./yaml_config.hpp"

// Direct Form I digital IIR filter
// (https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_1)
class DigitalFilter {
public:
  DigitalFilter(const Eigen::VectorXd &a, const Eigen::VectorXd &b, int dim = 1)
      : a_(a), b_(b) {
    VALIDATE_EQ(a.size(), b.size());
    resize(dim);
  }

  void resize(int dim) {
    x_.resize(a_.size(), dim);
    x_.setZero();

    y_.resize(a_.size(), dim);
    y_.setZero();
  }

  void reset(const Eigen::VectorXd &x0) {
    x_.rowwise() = x0.transpose();
    y_.rowwise() = x0.transpose();
  }

  void filter(const Eigen::VectorXd &obs, Eigen::VectorXd *out) {
    // FIXME(stuarta): perf: use an inplace circShift here, rather than copy.
    for (int i = 0; i < x_.cols(); ++i) {
      x_.col(i).tail(x_.rows() - 1) = x_.col(i).head(x_.rows() - 1).eval();
      y_.col(i).tail(y_.rows() - 1) = y_.col(i).head(y_.rows() - 1).eval();
    }
    x_.row(0) = obs.transpose();
    for (int i = 0; i < x_.cols(); ++i) {
      y_(0, i) = x_.col(i).dot(b_) -
                 y_.col(i).tail(a_.size() - 1).dot(a_.tail(a_.size() - 1));
    }
    *out = y_.row(0).transpose() / a_(0);
  }

private:
  Eigen::MatrixXd x_;
  Eigen::MatrixXd y_;
  const Eigen::VectorXd a_;
  const Eigen::VectorXd b_;
};

std::optional<DigitalFilter> loadFilter(YAML::Node filter) {
  return {
      {filter['A'].as<Eigen::VectorXd>(), filter['B'].as<Eigen::VectorXd>()}};
}

std::optional<DigitalFilter> loadFilter(std::string filename,
                                        std::string filter_name) {
  auto root = YAML::LoadFile(filename);
  auto filter = root[filter_name];
  return loadFilter(filter);
}

class DifferenceFilter {
public:
  explicit DifferenceFilter(int dim = 1) : prev_(dim) { prev_.setZero(); }

  void reset(const Eigen::VectorXd &obs) { prev_ = obs; }

  void filter(const Eigen::VectorXd &obs, Eigen::VectorXd *diff) {
    *diff = obs - prev_;
    prev_ = obs;
  }

private:
  Eigen::VectorXd prev_;
};

// FIXME(stuarta): refactor to DRY wrt DifferenceFilter
class SlewLimiter {
public:
  explicit SlewLimiter(double max_delta, int dim = 1)
      : max_delta_(max_delta), prev_(dim) {
    prev_.setZero();
  }

  void reset(const Eigen::Ref<Eigen::VectorXd> &obs) { prev_ = obs; }

  void filter(const Eigen::Ref<Eigen::VectorXd> &obs,
              Eigen::Ref<Eigen::VectorXd> out) {
    out = prev_ + (obs - prev_).cwiseMax(-max_delta_).cwiseMin(max_delta_);
    prev_ = out;
  }

private:
  double max_delta_;
  Eigen::VectorXd prev_;
};

class MedianFilter {
public:
  explicit MedianFilter(int dims = 1) { resize(dims); }

  void resize(int dim) {
    x_.resize(dim, 3);
    x_.setZero();
  }

  void reset() { x_.setZero(); }

  // It's OK for obs and out to be the same vector...
  void filter(const Eigen::VectorXd &obs, Eigen::VectorXd *out) {
    VALIDATE(out->rows() == x_.rows());
    x_.leftCols(2) = x_.rightCols(2).eval();
    x_.rightCols(1) = obs;

    for (int r = 0; r < x_.rows(); r++) {
      (*out)(r) = median3(x_(r, 0), x_(r, 1), x_(r, 2));
    }
  }

private:
  Eigen::MatrixXd x_;
};
