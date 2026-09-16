// The metric skip list: a random permutation of the points plus the finger
// lists of every point. This header holds the class shell; the algorithms
// are member functions defined in walk.hpp, build_seq.hpp, query.hpp and
// build_par.hpp, which are included at the end.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "mskip/finger_list.hpp"
#include "mskip/types.hpp"

namespace mskip {

// Counters kept by the builders (docs/PLAN.md section 6, "instrumentation").
struct BuildStats {
  size_t walks = 0;          // random walks run (resumes count again; provisional points do not)
  size_t steps = 0;          // iterations over all walks
  size_t focus_moves = 0;    // align steps locating focus lists (advance mode)
  size_t pointer_moves = 0;  // align steps settling advance pointers
  size_t merges = 0;         // merge phases of the parallel build
  size_t layers = 0;         // control-forest layers over all merges
  size_t max_forest_depth = 0;  // deepest control forest of any merge
};

template <class Metric>
class MetricSkipList {
 public:
  using point_type = typename Metric::point_type;
  static_assert(std::is_same_v<decltype(Metric::slack), const dist_t>,
                "Metric needs `static constexpr dist_t slack` (see metric.hpp)");

  // Shuffles pts with `seed` (parlay::random_permutation: deterministic for a
  // given seed, whatever the number of workers) and keeps the permutation.
  // The size and time bounds hold for a random permutation; with untrusted
  // input, use a seed the input cannot anticipate.
  // Throws std::invalid_argument for alpha outside [1, kMaxAlpha] or too
  // many points for idx_t.
  MetricSkipList(parlay::sequence<point_type> pts, idx_t alpha, Metric m = Metric(), uint64_t seed = 0)
      : metric_(m), alpha_(alpha), n_(checked_size(pts.size())) {
    perm_ = parlay::random_permutation<idx_t>(n_, parlay::random(seed));
    init(std::move(pts));
  }

  // Uses the given permutation instead of shuffling: perm[i] is the original
  // index of s_i. For tests and experiments with non-random orders.
  MetricSkipList(parlay::sequence<point_type> pts, idx_t alpha, parlay::sequence<idx_t> perm, Metric m = Metric())
      : metric_(m), alpha_(alpha), n_(checked_size(pts.size())), perm_(std::move(perm)) {
    if (perm_.size() != n_) throw std::invalid_argument("mskip: permutation size");
    parlay::sequence<uint8_t> seen(n_, uint8_t{0});
    for (idx_t p : perm_) {
      if (p >= n_ || seen[p]) throw std::invalid_argument("mskip: not a permutation");
      seen[p] = 1;
    }
    init(std::move(pts));
  }

  // Alg. 5 (advance = true: focus lists found with advance pointers and
  // align, and the pointers are stored) or Alg. 2 (binary search, no
  // pointers): F_{n-1} down to F_0, one random walk each.
  void build_sequential(bool advance = true);
  // Alg. 6 (+ Sec. 5.5 when advance = true): divide and conquer with the
  // control forest. Ranges of at most seq_base points (never fewer than
  // alpha) form the sequential base case.
  void build_parallel(size_t seq_base = kDefaultSeqBase, bool advance = true);
  static constexpr size_t kDefaultSeqBase = 1024;
  bool has_advance() const { return advance_; }  // queries then use Alg. 4
  size_t unresolved() const {                    // advance pointers not yet exact; 0 after a build
    size_t total = 0;
    for (const auto& p : pending_) total += p.size();
    return total;
  }

  // Results are original indices (into the pts passed to the constructor).
  // Ties: the closest point of highest priority (lowest permutation index).
  idx_t nearest(const point_type& q) const;                               // throws std::out_of_range if n == 0
  parlay::sequence<idx_t> knn(const point_type& q, idx_t k) const;        // by distance, then priority
  parlay::sequence<idx_t> range(const point_type& q, dist_t delta) const;  // open ball d < delta, discovery order

  idx_t n() const { return n_; }
  idx_t alpha() const { return alpha_; }
  const Metric& metric() const { return metric_; }
  const point_type& point(idx_t i) const { return pts_[i]; }  // s_i
  // Distances are always evaluated as metric(owner or query, other point).
  dist_t dist(idx_t i, idx_t j) const { return metric_(pts_[i], pts_[j]); }
  dist_t dist_to(idx_t j, const point_type& q) const { return metric_(q, pts_[j]); }
  dist_t radius_slack() const { return 1 + Metric::slack; }  // factor on every search radius
  const parlay::sequence<idx_t>& permutation() const { return perm_; }
  const FingerLists& lists(idx_t i) const { return lists_[i]; }
  // Control point C[i]: where the walk that built F_i stopped (i itself when
  // s_i never had alpha successors), and the index of the focus list there.
  const parlay::sequence<idx_t>& control() const { return control_; }
  const parlay::sequence<idx_t>& control_list() const { return control_list_; }
  const BuildStats& stats() const { return stats_; }

 private:
  static idx_t checked_size(size_t n) {
    if (n > static_cast<size_t>(std::numeric_limits<idx_t>::max()) - kMaxAlpha)
      throw std::invalid_argument("mskip: too many points for idx_t");
    return static_cast<idx_t>(n);
  }
  void init(parlay::sequence<point_type> pts) {
    if (alpha_ < 1 || alpha_ > kMaxAlpha) throw std::invalid_argument("mskip: alpha must be in [1, 255]");
    pts_ = parlay::tabulate(n_, [&](size_t i) { return pts[perm_[i]]; });
    lists_ = parlay::sequence<FingerLists>(n_, FingerLists(alpha_));
    control_ = parlay::tabulate(n_, [](size_t i) { return static_cast<idx_t>(i); });
    control_list_ = parlay::sequence<idx_t>(n_, idx_t{0});
  }

  struct BuildPolicy;  // build_seq.hpp
  void reset(bool advance);
  void finish_stats();
  size_t build_point(idx_t i, idx_t r, idx_t settled_from);
  void provisional(idx_t i, idx_t r);
  template <class Nav>
  idx_t query_nearest(const point_type& q) const;  // query.hpp
  template <class Nav>
  void query_knn(const point_type& q, parlay::sequence<Entry>& K) const;
  template <class Nav>
  void query_range(const point_type& q, dist_t delta, parlay::sequence<idx_t>& out) const;
  void parallel_build(idx_t l, idx_t r);  // build_par.hpp
  void merge(idx_t l, idx_t m, idx_t r);
  size_t resume(idx_t i, idx_t m, idx_t r);
  void fixup(idx_t i, bool final);

  // Per-worker BuildStats, summed at the end. Each worker only ever updates
  // its own entry (looked up after every call that may fork), so this needs
  // no atomics under child- or continuation-stealing schedulers.
  struct alignas(64) WorkerCounters {
    size_t walks = 0, steps = 0, focus_moves = 0, pointer_moves = 0, merges = 0, layers = 0, max_depth = 0;
  };
  WorkerCounters& counters() { return counters_[parlay::worker_id()]; }

  Metric metric_;
  idx_t alpha_;
  idx_t n_;
  parlay::sequence<idx_t> perm_;       // perm_[i] = original index of s_i
  parlay::sequence<point_type> pts_;   // in permutation order
  parlay::sequence<FingerLists> lists_;
  parlay::sequence<idx_t> control_, control_list_;
  bool advance_ = false;
  parlay::sequence<parlay::sequence<uint32_t>> pending_;  // per point: slots of adv not yet exact
  BuildStats stats_;
  std::vector<WorkerCounters> counters_;
  size_t seq_base_ = kDefaultSeqBase;
};

}  // namespace mskip

#include "mskip/walk.hpp"
#include "mskip/build_seq.hpp"
#include "mskip/query.hpp"
#include "mskip/build_par.hpp"
