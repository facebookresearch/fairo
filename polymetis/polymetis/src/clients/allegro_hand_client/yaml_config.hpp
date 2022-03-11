// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "Eigen/Eigen"
#include "yaml-cpp/yaml.h"

namespace YAML {

template <> struct convert<Eigen::VectorXd> {
  static Node encode(const Eigen::VectorXd &rhs) {
    Node node;
    for (int i = 0; i < rhs.size(); i++) {
      node.push_back(rhs(i));
    }
    return node;
  }

  static bool decode(const Node &node,
                     Eigen::VectorXd &rhs) { // NOLINT(runtime/references)
    if (!node.IsSequence()) {
      spdlog::error("Tried to load a non-sequence node as a vector.");
      return false;
    }
    rhs.resize(node.size());
    for (int i = 0; i < rhs.size(); i++) {
      rhs(i) = node[i].as<double>();
    }
    return true;
  }
};

} // namespace YAML
