// The random walk shared by construction and every query (docs/PLAN.md
// section 3). A Policy owns the candidate set K; a Nav locates the focus list.
#pragma once

#include <algorithm>
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
// Locates F* = F_cur(r) by binary search, as in Alg. 2/3. Phase 3 adds a
// navigator based on advance pointers and align behind the same interface.
struct BinarySearchNav {
  idx_t focus(const FingerLists& F, dist_t r) const { return F.locate(r); }
  idx_t hint(const FingerLists&, idx_t, idx_t) const { return 0; }
  void hop(idx_t) const {}
};

// Policy concept:
//   dist_t radius() const;                          // radius of K
//   void   offer(idx_t j, dist_t dj, idx_t hint);   // the chosen nxt; K inserts it iff dj < radius()
//   void   stop(idx_t cur, idx_t k);                // the walk ended at focus list k of s_cur
//
// Invariant: K is exact for the prefix of the permutation up to cur, and nxt
// is chosen so that no point strictly between cur and nxt could improve K.
// Returns the number of iterations.
template <class Metric, class Policy, class Nav>
size_t random_walk(const MetricSkipList<Metric>& S, const typename Metric::point_type& q, Policy& K,
                   idx_t cur, Nav& nav) {
  const idx_t alpha = S.alpha();
  for (size_t steps = 1;; steps++) {
    const FingerLists& F = S.lists(cur);
    const dist_t dc = S.dist_to(cur, q);
    const dist_t thr = std::max(K.radius(), dc);  // "improves K or is closer than cur"
    const idx_t k = nav.focus(F, dc + thr);       // any such point lies within dc + thr of cur
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
