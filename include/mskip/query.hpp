// Queries: nearest neighbor (Alg. 3), k nearest neighbors and range search,
// all as instances of the random walk with a different candidate set.
#pragma once

#include <algorithm>
#include <cassert>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "mskip/metric_skip_list.hpp"
#include "mskip/walk.hpp"

namespace mskip {
namespace detail {

// K = {m}, the closest point seen so far (ties: highest priority).
struct NearestPolicy {
  idx_t m;
  dist_t dm;
  dist_t radius() const { return dm; }
  void offer(idx_t j, dist_t dj, idx_t) {
    if (dj < dm) {
      m = j;
      dm = dj;
    }
  }
  void stop(idx_t, idx_t) const {}
};

// K = the k closest points seen so far under the (dist, idx) order.
struct KnnPolicy {
  parlay::sequence<Entry>& K;
  dist_t rad;
  dist_t radius() const { return rad; }
  void offer(idx_t j, dist_t dj, idx_t) {
    if (!(dj < rad)) return;
    K[farthest(K.data(), static_cast<idx_t>(K.size()))] = Entry{j, dj};
    rad = 0;
    for (const Entry& x : K) rad = std::max(rad, x.dist);
  }
  void stop(idx_t, idx_t) const {}
};

// K = the open ball of radius delta; every offered point inside it is output.
struct RangePolicy {
  dist_t delta;
  parlay::sequence<idx_t>& out;
  dist_t radius() const { return delta; }
  void offer(idx_t j, dist_t dj, idx_t) {
    if (dj < delta) out.push_back(j);
  }
  void stop(idx_t, idx_t) const {}
};

}  // namespace detail

template <class Metric>
idx_t MetricSkipList<Metric>::nearest(const point_type& q) const {
  assert(n_ > 0);
  detail::NearestPolicy K{0, dist_to(0, q)};
  BinarySearchNav nav;
  random_walk(*this, q, K, 0, nav);
  return perm_[K.m];
}

template <class Metric>
parlay::sequence<idx_t> MetricSkipList<Metric>::knn(const point_type& q, idx_t k) const {
  k = std::min(k, n_);
  parlay::sequence<Entry> K = parlay::tabulate(k, [&](size_t e) {
    return Entry{static_cast<idx_t>(e), dist_to(static_cast<idx_t>(e), q)};
  });
  if (k > 0 && k < n_) {
    dist_t rad = 0;
    for (const Entry& x : K) rad = std::max(rad, x.dist);
    detail::KnnPolicy P{K, rad};
    BinarySearchNav nav;
    random_walk(*this, q, P, k - 1, nav);
  }
  std::sort(K.begin(), K.end(), [](const Entry& a, const Entry& b) { return farther(b, a); });
  return parlay::map(K, [&](const Entry& x) { return perm_[x.idx]; });
}

template <class Metric>
parlay::sequence<idx_t> MetricSkipList<Metric>::range(const point_type& q, dist_t delta) const {
  parlay::sequence<idx_t> out;
  if (n_ == 0) return out;
  if (dist_to(0, q) < delta) out.push_back(0);
  detail::RangePolicy P{delta, out};
  BinarySearchNav nav;
  random_walk(*this, q, P, 0, nav);
  for (idx_t& x : out) x = perm_[x];
  return out;
}

}  // namespace mskip
