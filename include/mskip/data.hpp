// Seeded point generators for tests and benchmarks.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "mskip/metric.hpp"

namespace mskip::data {

inline double unit(parlay::random& r, size_t i) {  // uniform in [0, 1)
  return static_cast<double>(r.ith_rand(i) >> 11) * 0x1.0p-53;
}

// Uniform in [0, 1)^D.
template <size_t D>
parlay::sequence<Point<D>> uniform(size_t n, uint64_t seed = 0) {
  parlay::random r(seed);
  return parlay::tabulate(n, [&](size_t i) {
    Point<D> p;
    for (size_t k = 0; k < D; k++) p[k] = static_cast<dist_t>(unit(r, i * D + k));
    return p;
  });
}

// `centers` Gaussian clusters with the given sigma; centers uniform in [0, 1)^D.
// Cluster c has n/centers points, so density varies with sigma.
template <size_t D>
parlay::sequence<Point<D>> gaussian_clusters(size_t n, size_t centers, double sigma, uint64_t seed = 0) {
  parlay::random r(seed);
  auto mu = uniform<D>(centers, seed ^ 0x9e3779b97f4a7c15ULL);
  return parlay::tabulate(n, [&](size_t i) {
    Point<D> p = mu[i % centers];
    for (size_t k = 0; k < D; k++) {  // Box-Muller
      const double u1 = 1.0 - unit(r, 2 * (i * D + k));
      const double u2 = unit(r, 2 * (i * D + k) + 1);
      p[k] += static_cast<dist_t>(sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
    return p;
  });
}

// Integer coordinates in [0, side)^D: many duplicate points and equal distances.
template <size_t D>
parlay::sequence<Point<D>> grid(size_t n, unsigned side, uint64_t seed = 0) {
  parlay::random r(seed);
  return parlay::tabulate(n, [&](size_t i) {
    Point<D> p;
    for (size_t k = 0; k < D; k++) p[k] = static_cast<dist_t>(r.ith_rand(i * D + k) % side);
    return p;
  });
}

// Points on one line through the origin, at uniform random offsets.
template <size_t D>
parlay::sequence<Point<D>> collinear(size_t n, uint64_t seed = 0) {
  parlay::random r(seed);
  Point<D> dir;
  for (size_t k = 0; k < D; k++) dir[k] = static_cast<dist_t>(unit(r, 1000003 + k));
  return parlay::tabulate(n, [&](size_t i) {
    Point<D> p;
    const dist_t t = static_cast<dist_t>(unit(r, i));
    for (size_t k = 0; k < D; k++) p[k] = t * dir[k];
    return p;
  });
}

// Every point of pts repeated `copies` times (consecutively).
template <size_t D>
parlay::sequence<Point<D>> with_duplicates(const parlay::sequence<Point<D>>& pts, size_t copies) {
  return parlay::tabulate(pts.size() * copies, [&](size_t i) { return pts[i / copies]; });
}

}  // namespace mskip::data
