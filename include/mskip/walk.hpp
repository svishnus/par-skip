// The random walk shared by construction and every query (docs/PLAN.md
// section 3). A Policy owns the candidate set K; a Nav locates the focus list.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "mskip/finger_list.hpp"
#include "mskip/types.hpp"

namespace mskip {

template <class Metric>
class MetricSkipList;

// Nav concept:
//   static constexpr bool kHints;                          // the hint needs each entry's pointer slot
//   idx_t focus(const FingerLists& F, dist_t r);           // index of F(r) in F
//   idx_t hint(const FingerLists& F, const ListBuf& L, idx_t e); // start index in F_j for entry e of the focus list L
//   void  hop(idx_t h);                                    // the walk moves to that entry's point
//
// Locates F* = F_cur(r) by binary search, as in Alg. 2/3.
struct BinarySearchNav {
  static constexpr bool kHints = false;
  idx_t focus(const FingerLists& F, dist_t r) const { return F.locate(r); }
  idx_t hint(const FingerLists&, const ListBuf&, idx_t) const { return 0; }
  void hop(idx_t) const {}
};

// Locates F* with the advance pointers and align (Alg. 4/5): k is the index
// of the focus list in F_cur; the pointer of the chosen entry becomes the k
// for the next focus point, and align corrects it to F(r). Only the entries
// of checkpoint lists and the evictor of every other complete list have a
// stored pointer; a materialized list carries, for each entry, the slot of
// its most recent one (for a tail entry: from the last complete list, as in
// the paper; 0 when the point has no complete list). Every stored pointer is
// a valid index of F_j, so align from it is correct and only its distance
// varies. moves counts align steps: the cost paid instead of a binary search
// per step.
struct AdvanceNav {
  static constexpr bool kHints = true;
  idx_t k = 0;
  size_t moves = 0;
  idx_t focus(const FingerLists& F, dist_t r) {
    const idx_t k2 = F.align(k, r);
    moves += k2 > k ? k2 - k : k - k2;
    k = k2;
    return k;
  }
  idx_t hint(const FingerLists& F, const ListBuf& L, idx_t e) const { return F.adv_at(L.src[e]); }
  void hop(idx_t h) { k = h; }
};

// Policy concept:
//   dist_t radius() const;                          // radius of K
//   void   offer(idx_t j, dist_t dj, idx_t hint);   // the chosen nxt; K inserts it iff dj < radius()
//   void   stop(idx_t cur, idx_t k);                // the walk ended at focus list k of s_cur
//
// Invariant: K is exact for the prefix of the permutation up to cur, and nxt
// is chosen so that no point strictly between cur and nxt could improve K.
// The search ball is enlarged by the metric's slack so that the argument
// holds for the computed distances. Every hop goes to an entry with a larger
// index than cur, so the walk ends after at most n iterations whatever the
// metric does. Returns the number of iterations.
template <class Metric, class Policy, class Nav>
size_t random_walk(const MetricSkipList<Metric>& S, const typename Metric::point_type& q, Policy& K,
                   idx_t cur, Nav& nav) {
  const idx_t alpha = S.alpha();
  ListBuf L;  // the focus list, materialized once per step
  for (size_t steps = 1;; steps++) {
    const FingerLists& F = S.lists(cur);
    const dist_t dc = S.dist_to(cur, q);
    const dist_t thr = std::max(K.radius(), dc);  // "improves K or is closer than cur"
    const idx_t k = nav.focus(F, (dc + thr) * S.radius_slack());  // any such point lies within dc + thr of cur
    F.materialize<Nav::kHints>(k, L);
    idx_t e = 0;
    dist_t dj = 0;
    for (; e < L.size; e++) {  // first (highest-priority) entry with the property
      dj = S.dist_to(L.ent[e].idx, q);
      if (dj < thr) break;
    }
    idx_t h;
    if (e < L.size) {
      h = nav.hint(F, L, e);
      K.offer(L.ent[e].idx, dj, h);
    } else if (L.size < alpha) {  // tail list: it holds every candidate, none qualifies
      K.stop(cur, k);
      return steps;
    } else {  // walk outward to the lowest-priority entry
      e = L.size - 1;
      h = nav.hint(F, L, e);
    }
    nav.hop(h);
    cur = L.ent[e].idx;
  }
}

}  // namespace mskip
