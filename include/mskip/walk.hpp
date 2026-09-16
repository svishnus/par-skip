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
//   idx_t focus(const FingerLists& F, dist_t r);        // index of F(r) in F
//   idx_t hint(const FingerLists& F, idx_t k, idx_t e); // start index in F_j for entry e of list k
//   void  hop(idx_t h);                                 // the walk moves to that entry's point
//
// Locates F* = F_cur(r) by binary search, as in Alg. 2/3.
struct BinarySearchNav {
  idx_t focus(const FingerLists& F, dist_t r) const { return F.locate(r); }
  idx_t hint(const FingerLists&, idx_t, idx_t) const { return 0; }
  void hop(idx_t) const {}
};

// Locates F* with the advance pointers and align (Alg. 4/5): k is the index
// of the focus list in F_cur; the pointer of the chosen entry becomes the k
// for the next focus point, and align corrects it to F(r). Tail lists carry
// no pointers; their entries are a subset of the last complete list, whose
// pointer is used instead (0 when the point has no complete list). moves
// counts align steps; it is the cost the pointers save over binary search.
struct AdvanceNav {
  idx_t k = 0;
  size_t moves = 0;
  idx_t focus(const FingerLists& F, dist_t r) {
    const idx_t k2 = F.align(k, r);
    moves += k2 > k ? k2 - k : k - k2;
    k = k2;
    return k;
  }
  idx_t hint(const FingerLists& F, idx_t list, idx_t e) const {
    if (F.is_complete(list)) return F.adv_begin(list)[e];
    if (F.num_complete() == 0) return 0;
    const idx_t j = F.begin(list)[e].idx;
    const idx_t last = F.num_complete() - 1;
    const Entry* L = F.begin(last);
    for (idx_t t = 0; t < F.alpha; t++)
      if (L[t].idx == j) return F.adv_begin(last)[t];
    assert(false && "tail entry missing from the last complete list");
    return 0;
  }
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
  for (size_t steps = 1;; steps++) {
    const FingerLists& F = S.lists(cur);
    const dist_t dc = S.dist_to(cur, q);
    const dist_t thr = std::max(K.radius(), dc);  // "improves K or is closer than cur"
    const idx_t k = nav.focus(F, (dc + thr) * S.radius_slack());  // any such point lies within dc + thr of cur
    const Entry* L = F.begin(k);
    const idx_t sz = F.size[k];
    idx_t e = 0;
    dist_t dj = 0;
    for (; e < sz; e++) {  // first (highest-priority) entry with the property
      dj = S.dist_to(L[e].idx, q);
      if (dj < thr) break;
    }
    idx_t h;
    if (e < sz) {
      h = nav.hint(F, k, e);
      K.offer(L[e].idx, dj, h);
    } else if (sz < alpha) {  // tail list: it holds every candidate, none qualifies
      K.stop(cur, k);
      return steps;
    } else {  // walk outward to the lowest-priority entry
      e = sz - 1;
      h = nav.hint(F, k, e);
    }
    nav.hop(h);
    cur = L[e].idx;
  }
}

}  // namespace mskip
