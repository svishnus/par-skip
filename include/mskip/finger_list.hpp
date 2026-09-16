// Finger lists of one point: the alpha-stride storage shared by every builder.
//
// F_i is a sequence of lists with non-increasing radius. List k holds the
// alpha highest-priority (smallest-index) points within radius[k] of s_i, or
// all of them when there are fewer than alpha (a tail list). docs/PLAN.md
// section 1 states the invariants and the tie rule that every builder shares.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

#include <parlay/sequence.h>
#include <parlay/slice.h>

#include "mskip/types.hpp"

namespace mskip {

// size[] is a uint8_t and several routines keep one list in a stack buffer.
constexpr idx_t kMaxAlpha = 255;

// One member of a finger list of s_i.
struct Entry {
  idx_t idx;    // permutation position of the point; always > i
  dist_t dist;  // d(s_i, s_idx), cached
};

inline bool operator==(const Entry& a, const Entry& b) { return a.idx == b.idx && a.dist == b.dist; }
inline bool operator!=(const Entry& a, const Entry& b) { return !(a == b); }

// The tie rule: a is farther than b iff (a.dist, a.idx) > (b.dist, b.idx).
// "The farthest entry" of a list is the largest under this order, so every
// list is the alpha smallest entries of its prefix of the permutation and the
// structure is a pure function of the permutation.
inline bool farther(const Entry& a, const Entry& b) {
  return a.dist > b.dist || (a.dist == b.dist && a.idx > b.idx);
}

// Position of the farthest entry in [first, first + sz); sz > 0.
inline idx_t farthest(const Entry* first, idx_t sz) {
  assert(sz > 0);
  idx_t f = 0;
  for (idx_t e = 1; e < sz; e++)
    if (farther(first[e], first[f])) f = e;
  return f;
}

// All finger lists of one point. List k occupies
// entries[k*alpha, k*alpha + size[k]) and is sorted by idx ascending. Lists
// [0, n_complete) are complete (size == alpha); the rest is the tail, with
// strictly decreasing sizes ending in the empty list of radius 0.
struct FingerLists {
  idx_t alpha = 0;
  idx_t n_complete = 0;
  bool has_adv = false;  // advance pointers are maintained (Phase 3)
  parlay::sequence<dist_t> radius;   // max dist in list k, 0 if empty; non-increasing
  parlay::sequence<uint8_t> size;    // size[k] <= alpha
  parlay::sequence<Entry> entries;   // alpha-stride; padding slots are zero
  parlay::sequence<idx_t> adv;       // alpha-stride, parallel to entries, only when has_adv:
                                     // adv[k*alpha+e] is the index of F_j(radius[k]) in F_j
                                     // for j = entries[k*alpha+e].idx. Tail lists carry none.

  FingerLists() = default;
  explicit FingerLists(idx_t a, bool with_adv = false) : alpha(a), has_adv(with_adv) {
    assert(a >= 1 && a <= kMaxAlpha);
  }

  idx_t num_lists() const { return static_cast<idx_t>(radius.size()); }
  idx_t num_complete() const { return n_complete; }
  bool is_complete(idx_t k) const { return k < n_complete; }
  size_t slot(idx_t k, idx_t e) const { return static_cast<size_t>(k) * alpha + e; }

  const Entry* begin(idx_t k) const { return entries.data() + slot(k, 0); }
  parlay::slice<const Entry*, const Entry*> list(idx_t k) const {
    const Entry* b = begin(k);
    return parlay::make_slice(b, b + size[k]);
  }
  const idx_t* adv_begin(idx_t k) const { return adv.data() + slot(k, 0); }
  idx_t* adv_begin(idx_t k) { return adv.data() + slot(k, 0); }

  // F_i(r): index of the first list with radius <= r (the paper's Table 2
  // and Alg. 2; its Sec. 3.3 text says "last", which is a slip), by binary
  // search over the non-increasing radius array. The last list is empty with
  // radius 0, so the result is valid for every r >= 0.
  idx_t locate(dist_t r) const {
    assert(!radius.empty() && r >= 0);
    auto it = std::partition_point(radius.begin(), radius.end(), [r](dist_t x) { return x > r; });
    return static_cast<idx_t>(it - radius.begin());
  }

  // Same result as locate(r), reached by walking up or down from list k
  // (align in Alg. 4). Cheap when k is already close.
  idx_t align(idx_t k, dist_t r) const {
    assert(k < num_lists() && r >= 0);
    while (k > 0 && radius[k - 1] <= r) k--;
    while (radius[k] > r) k++;
    return k;
  }

  // Appends a list. Entries must be sorted by idx ascending and sz <= alpha;
  // a complete list may only follow complete lists.
  void push_back(const Entry* first, idx_t sz) {
    assert(sz <= alpha);
    assert(sz < alpha || n_complete == num_lists());
    dist_t rad = 0;
    for (idx_t e = 0; e < sz; e++) {
      assert(e == 0 || first[e - 1].idx < first[e].idx);
      rad = std::max(rad, first[e].dist);
    }
    radius.push_back(rad);
    size.push_back(static_cast<uint8_t>(sz));
    entries.append(first, first + sz);
    entries.append(alpha - sz, Entry{0, 0});
    if (has_adv) adv.append(alpha, idx_t{0});
    if (sz == alpha) n_complete++;
  }

  // Drops every tail list (size < alpha).
  void truncate_tail() {
    radius.resize(n_complete);
    size.resize(n_complete);
    entries.resize(static_cast<size_t>(n_complete) * alpha);
    if (has_adv) adv.resize(static_cast<size_t>(n_complete) * alpha);
  }

  // From the last list, repeatedly removes the farthest entry (tie rule) and
  // appends the result, until the empty list has been appended. With no lists
  // at all, appends just the empty list.
  void build_tail() {
    if (num_lists() == 0) {
      static constexpr Entry none{0, 0};
      push_back(&none, 0);
      return;
    }
    idx_t sz = size.back();
    Entry buf[kMaxAlpha];  // push_back may reallocate entries: work on a copy
    std::copy(begin(num_lists() - 1), begin(num_lists() - 1) + sz, buf);
    while (sz > 0) {
      const idx_t f = farthest(buf, sz);
      std::copy(buf + f + 1, buf + sz, buf + f);
      sz--;
      push_back(buf, sz);
    }
  }

  void clear() {
    radius.clear();
    size.clear();
    entries.clear();
    adv.clear();
    n_complete = 0;
  }

  // Structural invariants checkable without the metric: sizes, the
  // complete/tail split, non-increasing radii equal to the max cached
  // distance, idx sorted and > owner, and every tail list derived from its
  // predecessor by removing the farthest entry. Returns "" when they hold.
  std::string validate(idx_t owner) const {
    auto fail = [&](const std::string& what, idx_t k) {
      return "F_" + std::to_string(owner) + " list " + std::to_string(k) + ": " + what;
    };
    if (num_lists() == 0) return fail("no lists", 0);
    if (size.size() != radius.size() || entries.size() != radius.size() * alpha) return fail("array sizes", 0);
    if (has_adv && adv.size() != entries.size()) return fail("adv size", 0);
    idx_t complete = 0;
    for (idx_t k = 0; k < num_lists(); k++) {
      const idx_t sz = size[k];
      if (sz > alpha) return fail("size > alpha", k);
      if (sz == alpha) {
        if (k != complete) return fail("complete list after a tail list", k);
        complete++;
      }
      dist_t rad = 0;
      for (idx_t e = 0; e < sz; e++) {
        const Entry& x = begin(k)[e];
        if (x.idx <= owner) return fail("idx <= owner", k);
        if (e > 0 && !(begin(k)[e - 1].idx < x.idx)) return fail("not sorted by idx", k);
        if (x.dist < 0) return fail("negative dist", k);
        rad = std::max(rad, x.dist);
      }
      if (radius[k] != rad) return fail("radius != max dist", k);
      if (k > 0 && radius[k] > radius[k - 1]) return fail("radius increases", k);
      if (k > 0) {
        // Every list after the first is its predecessor minus the farthest
        // entry, plus (complete lists only) one evictor of lower priority
        // than everything before it and closer than the old radius.
        const idx_t psz = size[k - 1];
        const idx_t kept = sz == alpha ? alpha - 1 : sz;
        if (kept + 1 != psz) return fail("size does not follow from predecessor", k);
        const idx_t f = farthest(begin(k - 1), psz);
        for (idx_t e = 0, p = 0; e < kept; e++, p++) {
          if (p == f) p++;
          if (begin(k)[e] != begin(k - 1)[p]) return fail("not predecessor minus farthest", k);
        }
        if (sz == alpha) {
          const Entry& ev = begin(k)[alpha - 1];
          if (!(ev.dist < radius[k - 1])) return fail("evictor not closer than old radius", k);
          if (!(ev.idx > begin(k - 1)[psz - 1].idx)) return fail("evictor not of lowest priority", k);
        }
      }
    }
    if (complete != n_complete) return fail("n_complete", complete);
    if (size.back() != 0) return fail("last list not empty", num_lists() - 1);
    return "";
  }
};

// Bitwise comparison of the parts every builder must agree on. with_adv also
// compares the advance pointers of complete lists.
inline bool same_lists(const FingerLists& a, const FingerLists& b, bool with_adv = false) {
  if (a.alpha != b.alpha || a.num_lists() != b.num_lists() || a.n_complete != b.n_complete) return false;
  for (idx_t k = 0; k < a.num_lists(); k++) {
    if (a.radius[k] != b.radius[k] || a.size[k] != b.size[k]) return false;
    for (idx_t e = 0; e < a.size[k]; e++)
      if (a.begin(k)[e] != b.begin(k)[e]) return false;
    if (with_adv && k < a.n_complete)
      for (idx_t e = 0; e < a.alpha; e++)
        if (a.adv_begin(k)[e] != b.adv_begin(k)[e]) return false;
  }
  return true;
}

}  // namespace mskip
