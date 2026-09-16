// Parallel construction (Alg. 6, docs/PLAN.md section 6): divide and conquer
// over the permutation. After both halves of [l, r] are built against their
// own ranges, the left half is completed against [m+1, r] by resuming each
// blocked random walk at its control point C[i]. Walks that resume from the
// same control point depend on it having been resumed first, so the left
// half is processed layer by layer over the control forest.
//
// Advance pointers are not maintained by the parallel build yet (Sec. 5.5,
// next phase): it always builds the binary-search structure.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "mskip/build_seq.hpp"
#include "mskip/metric_skip_list.hpp"

namespace mskip {

template <class Metric>
void MetricSkipList<Metric>::build_parallel(size_t seq_base, bool /*advance*/) {
  reset(false);
  if (n_ == 0) return;
  seq_base_ = std::max<size_t>(seq_base, alpha_);
  parallel_build(0, n_ - 1);
  finish_stats();
}

// Builds every F_i, i in [l, r], against s_{i+1..r}, leaving C[i] at the
// point where the walk of s_i is blocked. Ranges of at most seq_base points
// are built sequentially (the same computation, without the merge overhead).
template <class Metric>
void MetricSkipList<Metric>::parallel_build(idx_t l, idx_t r) {
  if (static_cast<size_t>(r - l) + 1 <= seq_base_) {
    WorkerCounters& c = counters_[parlay::worker_id()];
    for (idx_t i = r + 1; i-- > l;) {
      c.steps += build_point(i, r, i + 1);
      c.walks++;
    }
    return;
  }
  const idx_t m = l + (r - l) / 2;
  parlay::par_do([&] { parallel_build(l, m); }, [&] { parallel_build(m + 1, r); });
  merge(l, m, r);
}

// Completes F_i for i in [l, m] against s_{m+1..r}. The control forest has an
// edge i -> C[i] for every non-root; a layer's walks read only F_{C[i]}
// (finished in an earlier layer) and right-half lists, and write disjoint
// F_i, so a layer runs in parallel without synchronization.
template <class Metric>
void MetricSkipList<Metric>::merge(idx_t l, idx_t m, idx_t r) {
  const size_t cnt = static_cast<size_t>(m - l) + 1;
  auto nodes = parlay::delayed_tabulate(cnt, [l](size_t t) { return static_cast<idx_t>(l + t); });
  auto frontier = parlay::filter(nodes, [&](idx_t i) { return C_[i] == i; });
  auto nonroots = parlay::filter(nodes, [&](idx_t i) { return C_[i] != i; });
  auto keyed = parlay::delayed_map(nonroots, [&](idx_t i) { return std::pair<idx_t, idx_t>(C_[i] - l, i); });
  auto sorted = parlay::counting_sort_by_keys(keyed, cnt);
  const parlay::sequence<idx_t>& children = sorted.first;  // grouped by parent
  const parlay::sequence<size_t>& offsets = sorted.second;
  size_t depth = 0;
  while (!frontier.empty()) {
    depth++;
    parlay::parallel_for(0, frontier.size(), [&](size_t t) {
      WorkerCounters& c = counters_[parlay::worker_id()];
      c.steps += resume(frontier[t], m, r);
      c.walks++;
    }, 1);
    frontier = parlay::flatten(parlay::map(frontier, [&](idx_t f) {
      return children.cut(offsets[f - l], offsets[f - l + 1]);
    }));
  }
  WorkerCounters& c = counters_[parlay::worker_id()];
  c.merges++;
  c.layers += depth;
  c.max_depth = std::max(c.max_depth, depth);
}

// Resumes the walk of s_i at C[i] against the range now ending at r. A point
// without a complete list (C[i] == i) starts from scratch, or stays
// provisional if it still has fewer than alpha successors.
template <class Metric>
size_t MetricSkipList<Metric>::resume(idx_t i, idx_t m, idx_t r) {
  if (C_[i] == i) return build_point(i, r, m + 1);
  FingerLists& F = lists_[i];
  F.truncate_tail();
  BuildPolicy K{*this, i, F, m + 1, true, nullptr};
  BinarySearchNav nav;
  return random_walk(*this, pts_[i], K, C_[i], nav);
}

}  // namespace mskip
