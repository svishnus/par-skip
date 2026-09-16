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
//     Hamming, ...). A slack is a relative bound, so it must hold for every
//     magnitude the metric can return: subnormal intermediate values (e.g.
//     squared differences below 2^-126 in float) lose relative precision
//     and would need a much larger one, which is why the metrics below
//     accumulate in double.
#pragma once

#include <array>
#include <cstddef>
#include <cmath>
#include <limits>

#include "mskip/types.hpp"

namespace mskip {

template <size_t D>
using Point = std::array<dist_t, D>;

constexpr dist_t kEps = std::numeric_limits<dist_t>::epsilon();

// The metrics below compute in double from float coordinates: differences
// and their squares are then exact or nearly so for every float input (no
// subnormal squares), and the only rounding that matters is the final one
// to float, at most eps/2 relative. The triangle inequality on computed
// values then holds within (1 + eps/2)^2 / (1 - eps/2), and the ball radius
// itself is rounded twice (sum, product), so a slack of 4 eps leaves a
// factor of 2 in reserve.
constexpr dist_t kFloatSlack = 4 * kEps;

template <size_t D>
struct L2 {
  using point_type = Point<D>;
  static constexpr dist_t slack = kFloatSlack;
  dist_t operator()(const point_type& a, const point_type& b) const {
    double s = 0;
    for (size_t k = 0; k < D; k++) {
      const double d = static_cast<double>(a[k]) - b[k];
      s += d * d;
    }
    return static_cast<dist_t>(std::sqrt(s));
  }
};

template <size_t D>
struct L1 {
  using point_type = Point<D>;
  static constexpr dist_t slack = kFloatSlack;
  dist_t operator()(const point_type& a, const point_type& b) const {
    double s = 0;
    for (size_t k = 0; k < D; k++) s += std::fabs(static_cast<double>(a[k]) - b[k]);
    return static_cast<dist_t>(s);
  }
};

template <size_t D>
struct Linf {
  using point_type = Point<D>;
  static constexpr dist_t slack = kFloatSlack;
  dist_t operator()(const point_type& a, const point_type& b) const {
    double m = 0;
    for (size_t k = 0; k < D; k++) m = std::max(m, std::fabs(static_cast<double>(a[k]) - b[k]));
    return static_cast<dist_t>(m);
  }
};

}  // namespace mskip
