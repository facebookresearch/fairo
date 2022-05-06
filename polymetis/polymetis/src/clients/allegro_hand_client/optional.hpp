// (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if __cplusplus < 201703L

#include <experimental/optional>

namespace std {
template <typename C> using optional = std::experimental::optional<C>;
}

#endif
