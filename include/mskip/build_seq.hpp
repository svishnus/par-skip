// Sequential construction (Alg. 2 / Alg. 5): F_i is built by one random walk
// that queries s_i against the already finished F_{i+1..n-1}. The policy and
// helpers here are shared with the parallel build.
#pragma once

#include <cstddef>
#include <cstdint>

#include <parlay/sequence.h>

#include "mskip/metric_skip_list.hpp"
#include "mskip/walk.hpp"

namespace mskip {

// K is the last list of F_i. offer(j) with d(s_i, s_j) < radius appends the
// list with the farthest entry replaced by j (j has the lowest priority seen
// so far, so appending keeps the idx order). stop() records the control point
// and builds the tail.
//
// With advance pointers (F.has_adv), every entry of the new list gets its
// pointer F_j(radius) by aligning in F_j from a nearby start: the previous
// list's pointer for entries kept from it ("push-down"), or the pointer the
// focus list held for the evictor ("shift focus"), as in Alg. 5. A target
// j < settled_from may be under construction by another worker, so its
// pointer is deferred: it keeps the start value and its slot is recorded in
// `pending`. A pointer that lands in a tail list of a target that is not
// final (its lists still grow at a later level) is recorded too. The parallel
// build re-aligns pending slots once the targets are finished.
template <class Metric>
struct MetricSkipList<Metric>::BuildPolicy {
  MetricSkipList& S;
  idx_t i;
  FingerLists& F;
  idx_t settled_from;  // targets >= settled_from may be read now
  bool final;          // every readable target is complete for good
  parlay::sequence<uint32_t>* pending;
  size_t moves = 0;  // align steps spent on pointers

  dist_t radius() const { return F.radius.back(); }

  idx_t settle(idx_t j, idx_t from, dist_t rho, size_t slot) {
    if (j < settled_from) {
      pending->push_back(static_cast<uint32_t>(slot));
      return from;
    }
    const FingerLists& Fj = S.lists_[j];
    const idx_t p = Fj.align(from, rho);
    moves += p > from ? p - from : from - p;
    if (!final && p >= Fj.num_complete()) pending->push_back(static_cast<uint32_t>(slot));
    return p;
  }

  void offer(idx_t j, dist_t dj, idx_t hint) {
    if (!(dj < F.radius.back())) return;
    const idx_t a = S.alpha_;
    const idx_t k = F.num_lists() - 1;
    const Entry* last = F.begin(k);
    const idx_t f = farthest(last, a);
    Entry buf[kMaxAlpha];
    for (idx_t e = 0, m = 0; e < a; e++)
      if (e != f) buf[m++] = last[e];
    buf[a - 1] = Entry{j, dj};
    F.push_back(buf, a);
    if (F.has_adv) {
      const dist_t rho = F.radius.back();
      const idx_t* prev = F.adv_begin(k);
      idx_t* now = F.adv_begin(k + 1);
      for (idx_t e = 0; e + 1 < a; e++) now[e] = settle(buf[e].idx, prev[e < f ? e : e + 1], rho, F.slot(k + 1, e));
      now[a - 1] = settle(j, hint, rho, F.slot(k + 1, a - 1));
    }
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
  if (advance_) pending_[i].clear();
}

// Builds F_i from scratch against s_{i+1..r} (Build-Finger in Alg. 2/5).
// Targets >= settled_from are finished and may be read for pointers.
// Returns the length of the walk.
template <class Metric>
size_t MetricSkipList<Metric>::build_point(idx_t i, idx_t r, idx_t settled_from) {
  if (i + alpha_ > r) {
    provisional(i, r);
    return 0;
  }
  FingerLists& F = lists_[i];
  F.clear();
  Entry buf[kMaxAlpha];
  for (idx_t e = 0; e < alpha_; e++) buf[e] = Entry{i + 1 + e, dist(i, i + 1 + e)};
  F.push_back(buf, alpha_);
  if (!advance_) {
    BuildPolicy K{*this, i, F, settled_from, true, nullptr};
    BinarySearchNav nav;
    return random_walk(*this, pts_[i], K, i + alpha_, nav);
  }
  pending_[i].clear();
  BuildPolicy K{*this, i, F, settled_from, r == n_ - 1, &pending_[i]};
  idx_t* adv0 = F.adv_begin(0);
  for (idx_t e = 0; e < alpha_; e++) adv0[e] = K.settle(buf[e].idx, 0, F.radius[0], F.slot(0, e));
  AdvanceNav nav;  // k = 0: the first list of s_cur
  const size_t steps = random_walk(*this, pts_[i], K, i + alpha_, nav);
  WorkerCounters& c = counters_[parlay::worker_id()];
  c.focus_moves += nav.moves;
  c.pointer_moves += K.moves;
  return steps;
}

// Resets every list for a build in the given mode.
template <class Metric>
void MetricSkipList<Metric>::reset(bool advance) {
  stats_ = BuildStats();
  advance_ = advance;
  counters_.assign(parlay::num_workers(), WorkerCounters());
  if (advance && pending_.size() != n_) pending_ = parlay::sequence<parlay::sequence<uint32_t>>(n_);
  parlay::parallel_for(0, n_, [&](size_t i) {
    lists_[i] = FingerLists(alpha_, advance);
    C_[i] = static_cast<idx_t>(i);
    K_[i] = 0;
    if (advance) pending_[i].clear();
  });
}

template <class Metric>
void MetricSkipList<Metric>::finish_stats() {
  for (const WorkerCounters& c : counters_) {
    stats_.walks += c.walks;
    stats_.steps += c.steps;
    stats_.focus_moves += c.focus_moves;
    stats_.pointer_moves += c.pointer_moves;
    stats_.merges += c.merges;
    stats_.layers += c.layers;
    stats_.max_forest_depth = std::max(stats_.max_forest_depth, c.max_depth);
  }
}

template <class Metric>
void MetricSkipList<Metric>::build_sequential(bool advance) {
  reset(advance);
  WorkerCounters& c = counters_[parlay::worker_id()];
  for (idx_t i = n_; i-- > 0;) {
    c.steps += build_point(i, n_ - 1, i + 1);
    c.walks++;
  }
  finish_stats();
}

}  // namespace mskip
