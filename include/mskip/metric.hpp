// Point types and distance functors.
//
// A Metric is any type with
//   using point_type = ...;
//   dist_t operator()(const point_type&, const point_type&) const;
//   static constexpr dist_t slack;
// The skip list never looks inside a point, so any object with a metric can
// be plugged in. Requirements on the *computed* values:
//   * finite and non-negative, and exactly symmetric (d(a, b) == d(b, a));
//   * the triangle inequality up to rounding: d(a, c) <= (d(a, b) + d(b, c))
//     * (1 + slack). The walks enlarge their search balls by this factor,
//     which is all the exactness argument needs. With floating-point
//     coordinates the computed L2 distance can exceed the sum of two others
//     by an ulp, so slack = 0 is only right for exact metrics (integer L1,
//     Hamming, ...).
#pragma once

#include <array>
#include <cmath>
#include <limits>

#include "mskip/types.hpp"

namespace mskip {

template <int D>
using Point = std::array<dist_t, D>;

constexpr dist_t kEps = std::numeric_limits<dist_t>::epsilon();

// Relative error of the computed distance is about (D/2 + 2) eps for L2 and
// D eps for L1 (D roundings in the sum), eps for Linf; the slack must cover
// twice that plus the roundings of the ball radius itself.
template <int D>
struct L2 {
  using point_type = Point<D>;
  static constexpr dist_t slack = (2 * D + 8) * kEps;
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
  static constexpr dist_t slack = (2 * D + 8) * kEps;
  dist_t operator()(const point_type& a, const point_type& b) const {
    dist_t s = 0;
    for (int k = 0; k < D; k++) s += std::fabs(a[k] - b[k]);
    return s;
  }
};

template <int D>
struct Linf {
  using point_type = Point<D>;
  static constexpr dist_t slack = 8 * kEps;
  dist_t operator()(const point_type& a, const point_type& b) const {
    dist_t m = 0;
    for (int k = 0; k < D; k++) m = std::max(m, std::fabs(a[k] - b[k]));
    return m;
  }
};

}  // namespace mskip
