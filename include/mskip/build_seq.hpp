// Sequential construction (Alg. 2): F_i is built by one random walk that
// queries s_i against the already finished F_{i+1..n-1}. The same policy and
// helpers are reused by the parallel build.
#pragma once

#include <cstddef>

#include "mskip/metric_skip_list.hpp"
#include "mskip/walk.hpp"

namespace mskip {

// K is the last list of F_i. offer(j) with d(s_i, s_j) < radius appends the
// list with the farthest entry replaced by j (j has the lowest priority seen
// so far, so appending keeps the idx order). stop() records the control point
// and builds the tail.
template <class Metric>
struct MetricSkipList<Metric>::BuildPolicy {
  MetricSkipList& S;
  idx_t i;
  FingerLists& F;

  dist_t radius() const { return F.radius.back(); }

  void offer(idx_t j, dist_t dj, idx_t /*hint*/) {
    if (!(dj < F.radius.back())) return;
    const idx_t a = S.alpha_;
    const Entry* last = F.begin(F.num_lists() - 1);
    const idx_t f = farthest(last, a);
    Entry buf[kMaxAlpha];
    idx_t m = 0;
    for (idx_t e = 0; e < a; e++)
      if (e != f) buf[m++] = last[e];
    buf[m] = Entry{j, dj};
    F.push_back(buf, a);
  }

  void stop(idx_t cur, idx_t k) {
    S.C_[i] = cur;
    S.K_[i] = k;
    F.build_tail();
  }
};

// Tail-only lists over {i+1 .. r}: fewer than alpha points, exact by
// definition, no walk needed. C[i] = i marks that s_i has no complete list.
template <class Metric>
void MetricSkipList<Metric>::provisional(idx_t i, idx_t r) {
  FingerLists& F = lists_[i];
  F.clear();
  const idx_t t = r - i;
  if (t > 0) {
    Entry buf[kMaxAlpha];
    for (idx_t e = 0; e < t; e++) buf[e] = Entry{i + 1 + e, dist(i, i + 1 + e)};
    F.push_back(buf, t);
  }
  F.build_tail();
  C_[i] = i;
  K_[i] = 0;
}

// Builds F_i from scratch against s_{i+1..r} (Build-Finger in Alg. 2).
// Returns the length of the walk.
template <class Metric>
size_t MetricSkipList<Metric>::build_point(idx_t i, idx_t r) {
  if (i + alpha_ > r) {
    provisional(i, r);
    return 0;
  }
  FingerLists& F = lists_[i];
  F.clear();
  Entry buf[kMaxAlpha];
  for (idx_t e = 0; e < alpha_; e++) buf[e] = Entry{i + 1 + e, dist(i, i + 1 + e)};
  F.push_back(buf, alpha_);
  BuildPolicy K{*this, i, F};
  BinarySearchNav nav;
  return random_walk(*this, pts_[i], K, i + alpha_, nav);
}

template <class Metric>
void MetricSkipList<Metric>::build_sequential() {
  stats_ = BuildStats();
  for (idx_t i = n_; i-- > 0;) {
    stats_.steps += build_point(i, n_ - 1);
    stats_.walks++;
  }
}

}  // namespace mskip
