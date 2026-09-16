// Parallel construction (Alg. 6, docs/PLAN.md section 6): divide and conquer
// over the permutation. After both halves of [l, r] are built against their
// own ranges, the left half is completed against [m+1, r] by resuming each
// blocked random walk at its control point C[i]. Walks that resume from the
// same control point depend on it having been resumed first, so the left
// half is processed layer by layer over the control forest.
//
// With advance pointers (Sec. 5.5) the resumed walks navigate with the
// pointers and align like the sequential Alg. 5, and new pointers are
// aligned immediately when their target is in the finished right half.
// Pointers into the left half are deferred, and pointers that land in the
// tail of a target whose lists still grow are provisional; both are recorded
// per point and re-aligned after the merge (see fixup). This replaces the
// paper's advance tree: every pointer is touched once per level at most, and
// the walks never wait for a pointer. Every stored index stays valid because
// no F_j is read while it is rebuilt (a point rebuilt from provisional is a
// root, and roots read only right-half lists) and every rebuild ends with at
// least as many lists as before.
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
void MetricSkipList<Metric>::build_parallel(size_t seq_base, bool advance) {
  reset(advance);
  seq_base_ = std::max<size_t>(seq_base, alpha_);
  if (n_ > 0) parallel_build(0, n_ - 1);
  finish_stats();
  assert(unresolved() == 0);
}

// Builds every F_i, i in [l, r], against s_{i+1..r}, leaving C[i] at the
// point where the walk of s_i is blocked. Ranges of at most seq_base points
// are built sequentially (the same computation, without the merge overhead).
template <class Metric>
void MetricSkipList<Metric>::parallel_build(idx_t l, idx_t r) {
  if (static_cast<size_t>(r - l) + 1 <= seq_base_) {
    for (idx_t i = r + 1; i-- > l;) {
      const size_t steps = build_point(i, r, i + 1);
      WorkerCounters& c = counters();
      c.steps += steps;
      c.walks += steps > 0;
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
  auto frontier = parlay::filter(nodes, [&](idx_t i) { return control_[i] == i; });
  auto nonroots = parlay::filter(nodes, [&](idx_t i) { return control_[i] != i; });
  auto keyed = parlay::delayed_map(nonroots, [&](idx_t i) { return std::pair<idx_t, idx_t>(control_[i] - l, i); });
  // The roots are exactly the points without a complete list, i.e. the last
  // min(alpha, cnt) points of the left half.
  assert(frontier.size() <= alpha_);
  auto sorted = parlay::counting_sort_by_keys(keyed, cnt);
  const parlay::sequence<idx_t>& children = sorted.first;  // grouped by parent
  const parlay::sequence<size_t>& offsets = sorted.second;
  size_t depth = 0;
  while (!frontier.empty()) {
    depth++;
    parlay::parallel_for(0, frontier.size(), [&](size_t t) {
      const size_t steps = resume(frontier[t], m, r);
      WorkerCounters& c = counters();
      c.steps += steps;
      c.walks += steps > 0;
    }, 1);
    frontier = parlay::flatten(parlay::map(frontier, [&](idx_t f) {
      return children.cut(offsets[f - l], offsets[f - l + 1]);
    }));
  }
  if (advance_) {
    const bool targets_final = r == n_ - 1;
    parlay::parallel_for(0, cnt, [&](size_t t) { fixup(static_cast<idx_t>(l + t), targets_final); });
  }
  WorkerCounters& c = counters();
  c.merges++;
  c.layers += depth;
  c.max_depth = std::max(c.max_depth, depth);
}

// Resumes the walk of s_i at C[i] against the range now ending at r. A point
// without a complete list (C[i] == i) starts from scratch, or stays
// provisional if it still has fewer than alpha successors. Targets in the
// left half [l, m] may be under construction in this layer, so pointers into
// them are deferred (settled_from = m + 1).
template <class Metric>
size_t MetricSkipList<Metric>::resume(idx_t i, idx_t m, idx_t r) {
  if (control_[i] == i) return build_point(i, r, m + 1);
  FingerLists& F = lists_[i];
  F.truncate_tail();
  assert(F.num_complete() > 0);
  assert(control_list_[i] < lists_[control_[i]].num_lists());  // lists never shrink
  if (!advance_) {
    BuildPolicy K{*this, i, F, m + 1, true, nullptr};
    BinarySearchNav nav;
    return random_walk(*this, pts_[i], K, control_[i], nav);
  }
  BuildPolicy K{*this, i, F, m + 1, r == n_ - 1, &pending_[i]};
  AdvanceNav nav{control_list_[i]};  // the focus list where the walk was blocked
  const size_t steps = random_walk(*this, pts_[i], K, control_[i], nav);
  WorkerCounters& c = counters();
  c.focus_moves += nav.moves;
  c.pointer_moves += K.moves;
  c.pointers_deferred += K.deferred;
  return steps;
}

// Re-aligns the recorded pointers of F_i now that every target is finished
// for this level. Slots are recorded in creation order, so when the previous
// list also holds the target (every entry but the evictor, at the same
// position or one to the right) its pointer has been fixed already and is
// the push-down start: radii decrease, so the answer only moves down from
// it. Otherwise the recorded start is used. A pointer that resolves to a
// complete list is exact for good; one that lands in a tail stays recorded
// unless the level is final.
template <class Metric>
void MetricSkipList<Metric>::fixup(idx_t i, bool targets_final) {
  parlay::sequence<uint32_t>& pend = pending_[i];
  if (pend.empty()) return;
  FingerLists& F = lists_[i];
  size_t moves = 0, w = 0;
  for (uint32_t s : pend) {
    const idx_t k = s / alpha_;
    const idx_t e = s - k * alpha_;
    const idx_t j = F.entries[s].idx;
    const FingerLists& Fj = lists_[j];
    idx_t from = F.adv[s];
    if (k > 0 && e + 1 < alpha_) {  // kept from list k-1 (the evictor sits at alpha-1)
      const idx_t t = F.begin(k - 1)[e].idx == j ? e : e + 1;
      assert(F.begin(k - 1)[t].idx == j);
      from = F.adv_begin(k - 1)[t];
    }
    const idx_t p = Fj.align(from, F.radius[k]);
    moves += p > from ? p - from : from - p;
    F.adv[s] = p;
    if (BuildPolicy::still_pending(Fj, p, targets_final)) pend[w++] = s;
  }
  WorkerCounters& c = counters();
  c.pointer_moves += moves;
  c.pointers_refixed += pend.size();
  c.pointers_kept += w;
  pend.resize(w);
}

}  // namespace mskip
