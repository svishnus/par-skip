// Sequential construction (Alg. 2 / Alg. 5): F_i is built by one random walk
// that queries s_i against the already finished F_{i+1..n-1}. The policy and
// helpers here are shared with the parallel build.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <parlay/sequence.h>

#include "mskip/metric_skip_list.hpp"
#include "mskip/walk.hpp"

namespace mskip {

// K is the last list of F_i, kept materialized in `cur`. offer(j) with
// d(s_i, s_j) < radius appends the list with the farthest entry replaced by
// j (j has the lowest priority seen so far, so appending keeps the idx
// order). stop() records the control point and builds the tail.
//
// With advance pointers (F.has_adv), the new list's evictor gets its pointer
// F_j(radius) by aligning in F_j from the pointer the focus list held for it
// ("shift focus"); at a checkpoint list every kept entry gets one too, from
// its most recent stored pointer ("push-down"), as in Alg. 5 but only where
// a pointer is stored. A target j < settled_from may be under construction
// by another worker, so its pointer is deferred: it keeps the start value
// and its slot is recorded in `pending`. A pointer that lands in a tail list
// of a target that is not final (its lists still grow at a later level) is
// recorded too. The parallel build re-aligns pending slots once the targets
// are finished.
template <class Metric>
struct MetricSkipList<Metric>::BuildPolicy {
  MetricSkipList& S;
  idx_t i;
  FingerLists& F;
  idx_t settled_from;   // targets >= settled_from may be read now
  bool targets_final;   // every readable target is complete for good
  parlay::sequence<uint32_t>* pending;
  size_t moves = 0;     // align steps spent on pointers
  size_t deferred = 0;  // pointers left unaligned (target not readable)
  ListBuf cur;          // the last list of F

  BuildPolicy(MetricSkipList& s, idx_t owner, FingerLists& f, idx_t settled, bool final_targets,
              parlay::sequence<uint32_t>* pend)
      : S(s), i(owner), F(f), settled_from(settled), targets_final(final_targets), pending(pend) {}

  dist_t radius() const { return F.last_radius(); }

  // A pointer that lands in the tail of a target whose lists may still grow
  // is not exact yet.
  static bool still_pending(const FingerLists& Fj, idx_t p, bool targets_final) {
    return !targets_final && p >= Fj.num_complete();
  }

  idx_t settle(idx_t j, idx_t from, dist_t rho, size_t slot) {
    assert(slot <= std::numeric_limits<uint32_t>::max());
    if (j < settled_from) {
      pending->push_back(static_cast<uint32_t>(slot));
      deferred++;
      return from;
    }
    const FingerLists& Fj = S.lists_[j];
    const idx_t p = Fj.align(from, rho);
    moves += p > from ? p - from : from - p;
    if (still_pending(Fj, p, targets_final)) pending->push_back(static_cast<uint32_t>(slot));
    return p;
  }

  // Settles the pointer of entry e of the new list k from `from` and records
  // its slot in cur.
  void settle_slot(idx_t k, idx_t e, idx_t from) {
    const size_t slot = F.adv_slot(k, e);
    F.adv_at(slot) = settle(cur.ent[e].idx, from, F.radius(k), slot);
    cur.src[e] = static_cast<uint32_t>(slot);
  }

  void offer(idx_t j, dist_t dj, idx_t hint) {
    if (!(dj < F.last_radius())) return;
    const idx_t a = S.alpha_;
    const idx_t f = farthest(cur.ent, a);
    const idx_t k = F.num_lists();  // the new list
    if (F.has_adv && F.is_checkpoint(k)) {
      // the kept entries get pointers of their own: start from their most
      // recent ones (push-down), read before push_evict retags the slots
      idx_t from[kMaxAlpha];
      for (idx_t e = 0, p = 0; e + 1 < a; e++, p++) {
        if (p == f) p++;
        from[e] = F.adv_at(cur.src[p]);
      }
      F.push_evict(cur, f, Entry{j, dj});
      for (idx_t e = 0; e + 1 < a; e++) settle_slot(k, e, from[e]);
    } else {
      F.push_evict(cur, f, Entry{j, dj});
    }
    if (F.has_adv) settle_slot(k, a - 1, hint);
  }

  void stop(idx_t cur_pt, idx_t k) {
    S.control_[i] = cur_pt;
    S.control_list_[i] = k;
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
    F.push_first(buf, t);
  }
  F.build_tail();
  control_[i] = i;
  control_list_[i] = 0;
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
  if (!advance_) {
    BuildPolicy K(*this, i, F, settled_from, true, nullptr);
    for (idx_t e = 0; e < alpha_; e++) K.cur.ent[e] = Entry{i + 1 + e, dist(i, i + 1 + e)};
    K.cur.size = alpha_;
    F.push_first(K.cur.ent, alpha_);
    F.materialize<true>(0, K.cur);  // the slot tags
    BinarySearchNav nav;
    return random_walk(*this, pts_[i], K, i + alpha_, nav);
  }
  pending_[i].clear();
  BuildPolicy K(*this, i, F, settled_from, r == n_ - 1, &pending_[i]);
  for (idx_t e = 0; e < alpha_; e++) K.cur.ent[e] = Entry{i + 1 + e, dist(i, i + 1 + e)};
  K.cur.size = alpha_;
  F.push_first(K.cur.ent, alpha_);
  for (idx_t e = 0; e < alpha_; e++) K.settle_slot(0, e, 0);  // every pointer starts at F_j[0]
  AdvanceNav nav;  // k = 0: the first list of s_cur
  const size_t steps = random_walk(*this, pts_[i], K, i + alpha_, nav);
  WorkerCounters& c = counters();
  c.focus_moves += nav.moves;
  c.pointer_moves += K.moves;
  c.pointers_deferred += K.deferred;
  return steps;
}

// Stored lists to reserve for s_i: the expected number, alpha (ln (n - i) -
// ln alpha) + 1 (one per evictor plus the first list; the evictor count is
// a sum of independent Bernoullis, so its standard deviation is about the
// square root of its mean), plus two standard deviations, so that only a
// few percent of the points ever grow their buffer.
template <class Metric>
idx_t MetricSkipList<Metric>::reserve_lists(idx_t i) const {
  const idx_t left = n_ - i;  // points after s_i, plus one
  if (left <= alpha_) return 1;
  const double mean = alpha_ * (std::log(double(left)) - std::log(double(alpha_))) + 1;
  const double want = std::ceil(mean + 2 * std::sqrt(std::max(mean, 1.0)));
  return static_cast<idx_t>(std::min<double>(want, left - alpha_ + 1));
}

// Resets every list for a build in the given mode. The initial buffers are
// slices of one slab (see reserve_lists), so the allocator's rounding is not
// paid per point; a point that outgrows its slice moves to a buffer of its
// own, and the slice is abandoned until the next build.
template <class Metric>
void MetricSkipList<Metric>::reset(bool advance) {
  stats_ = BuildStats();
  built_ = false;
  advance_ = advance;
  counters_.assign(parlay::num_workers(), WorkerCounters());
  if (advance && pending_.size() != n_) pending_ = parlay::sequence<parlay::sequence<uint32_t>>(n_);
  const FingerLists proto(alpha_, advance, stride_);
  auto bytes = parlay::tabulate(n_, [&](size_t i) { return proto.bytes_for(reserve_lists(static_cast<idx_t>(i))); });
  auto scanned = parlay::scan(bytes);
  const parlay::sequence<size_t>& offset = scanned.first;
  parlay::parallel_for(0, n_, [&](size_t i) {  // drop every buffer before the slab they may live in
    lists_[i] = FingerLists(alpha_, advance, stride_);
    control_[i] = static_cast<idx_t>(i);
    control_list_[i] = 0;
    if (advance) pending_[i].clear();
  });
  slab_.mem = parlay::sequence<std::byte>();  // release the old slab before the new one is mapped
  slab_.mem = parlay::sequence<std::byte>::uninitialized(scanned.second);
  parlay::parallel_for(0, n_, [&](size_t i) {
    lists_[i].attach(slab_.mem.data() + offset[i], reserve_lists(static_cast<idx_t>(i)));
  });
}

template <class Metric>
void MetricSkipList<Metric>::finish_stats() {
  built_ = true;
  for (const WorkerCounters& c : counters_) {
    stats_.walks += c.walks;
    stats_.steps += c.steps;
    stats_.focus_moves += c.focus_moves;
    stats_.pointer_moves += c.pointer_moves;
    stats_.pointers_deferred += c.pointers_deferred;
    stats_.pointers_refixed += c.pointers_refixed;
    stats_.pointers_kept += c.pointers_kept;
    stats_.merges += c.merges;
    stats_.layers += c.layers;
    stats_.max_forest_depth = std::max(stats_.max_forest_depth, c.max_depth);
  }
}

template <class Metric>
void MetricSkipList<Metric>::build_sequential(bool advance) {
  reset(advance);
  for (idx_t i = n_; i-- > 0;) {
    const size_t steps = build_point(i, n_ - 1, i + 1);
    WorkerCounters& c = counters();
    c.steps += steps;
    c.walks += steps > 0;
  }
  finish_stats();
  assert(unresolved() == 0);
}

}  // namespace mskip
