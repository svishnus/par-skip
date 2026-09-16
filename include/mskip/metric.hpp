// Point types and distance functors.
//
// A Metric is any type with
//   using point_type = ...;
//   dist_t operator()(const point_type&, const point_type&) const;
// satisfying the metric axioms (only the triangle inequality is actually
// needed for correctness of the walks). The skip list never looks inside a
// point, so any object with a metric can be plugged in.
#pragma once

#include <array>
#include <cmath>

#include "mskip/types.hpp"

namespace mskip {

template <int D>
using Point = std::array<dist_t, D>;

template <int D>
struct L2 {
  using point_type = Point<D>;
  dist_t operator()(const point_type& a, const point_type& b) const {
    dist_t s = 0;
    for (int k = 0; k < D; k++) {
      dist_t d = a[k] - b[k];
      s += d * d;
    }
    return std::sqrt(s);
  }
};

template <int D>
struct L1 {
  using point_type = Point<D>;
  dist_t operator()(const point_type& a, const point_type& b) const {
    dist_t s = 0;
    for (int k = 0; k < D; k++) s += std::fabs(a[k] - b[k]);
    return s;
  }
};

template <int D>
struct Linf {
  using point_type = Point<D>;
  dist_t operator()(const point_type& a, const point_type& b) const {
    dist_t m = 0;
    for (int k = 0; k < D; k++) m = std::max(m, std::fabs(a[k] - b[k]));
    return m;
  }
};

}  // namespace mskip
