// Definition-literal builders and brute-force oracles for the tests. Nothing
// here is used by the library itself.
#pragma once

#include <algorithm>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "mskip/metric_skip_list.hpp"

namespace mskip {

// F_i straight from the definition: scan s_{i+1..n-1} in permutation order,
// replacing the farthest entry whenever a strictly closer point appears.
// O(n * alpha) per point.
template <class Metric>
FingerLists reference_lists(const MetricSkipList<Metric>& S, idx_t i) {
  const idx_t a = S.alpha();
  const idx_t n = S.n();
  FingerLists F(a);
  Entry buf[kMaxAlpha];
  const idx_t t = std::min<idx_t>(a, n - 1 - i);
  for (idx_t e = 0; e < t; e++) buf[e] = Entry{i + 1 + e, S.dist(i, i + 1 + e)};
  F.push_back(buf, t);
  if (t == a) {
    for (idx_t j = i + a + 1; j < n; j++) {
      const dist_t d = S.dist(i, j);
      if (d < F.radius.back()) {
        const idx_t f = farthest(buf, a);
        std::copy(buf + f + 1, buf + a, buf + f);
        buf[a - 1] = Entry{j, d};
        F.push_back(buf, a);
      }
    }
  }
  F.build_tail();
  return F;
}

// The semantic definition of F_i(r): the alpha highest-priority points among
// { j > i : d(s_i, s_j) <= r }, or all of them if there are fewer.
template <class Metric>
parlay::sequence<idx_t> ball_prefix(const MetricSkipList<Metric>& S, idx_t i, dist_t r) {
  parlay::sequence<idx_t> out;
  for (idx_t j = i + 1; j < S.n() && out.size() < S.alpha(); j++)
    if (S.dist(i, j) <= r) out.push_back(j);
  return out;
}

// Brute-force queries in permutation order; ties resolved like the walks
// (closest first, then highest priority).
template <class Metric>
parlay::sequence<Entry> brute_knn(const MetricSkipList<Metric>& S, const typename Metric::point_type& q,
                                  idx_t k) {
  auto all = parlay::tabulate(S.n(), [&](size_t i) {
    return Entry{static_cast<idx_t>(i), S.dist_to(static_cast<idx_t>(i), q)};
  });
  std::sort(all.begin(), all.end(), [](const Entry& a, const Entry& b) { return farther(b, a); });
  all.resize(std::min<size_t>(k, all.size()));
  return all;
}

template <class Metric>
parlay::sequence<idx_t> brute_range(const MetricSkipList<Metric>& S, const typename Metric::point_type& q,
                                    dist_t delta) {
  parlay::sequence<idx_t> out;
  for (idx_t i = 0; i < S.n(); i++)
    if (S.dist_to(i, q) < delta) out.push_back(i);
  return out;
}

}  // namespace mskip
