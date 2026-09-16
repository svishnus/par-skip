// Finger lists of one point: the compact storage shared by every builder.
//
// F_i is a sequence of lists with non-increasing radius. List k holds the
// alpha highest-priority (smallest-index) points within radius[k] of s_i, or
// all of them when there are fewer than alpha (a tail list). docs/PLAN.md
// section 1 states the invariants and the tie rule that every builder shares.
//
// Consecutive complete lists differ by one entry: list k is list k-1 with its
// farthest entry removed and an evictor of lower priority than everything
// else appended. So only list 0 is stored in full; every later complete list
// is stored as (evicted position, evictor), with a full copy every `stride`
// lists (a checkpoint) so that any list is materialized by replaying at most
// stride - 1 deltas. The tail lists are the last stored list minus its
// farthest entries: that list is kept once more, with the order in which its
// entries drop out, and only the tail's radii are stored, so that locate and
// align see one contiguous radius array. This is Θ(α ln n) per point instead
// of the Θ(α² ln n) of an alpha-stride layout.
//
// A materialized list is the live subset of its block's slots: the alpha
// checkpoint entries followed by the block's evictors, in idx order. A delta
// records the slot it kills (one byte), so replaying a block is a few bit
// operations on a live mask followed by one gather.
//
// Everything lives in one buffer per point, laid out for a capacity of `cap`
// stored lists (see layout()): a slice of a slab the owner hands out, or
// its own allocation once it outgrows the slice. The walk touches one
// object of 48 bytes plus the sections it needs.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include <parlay/alloc.h>

#include "mskip/types.hpp"

namespace mskip {

// Positions within a list are uint8_t and several routines keep one list in
// a stack buffer.
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

// One list, materialized: its entries in idx order and, per entry, the slot
// of its stored advance pointer (FingerLists::adv_at(src[e]); see adv_slot).
struct ListBuf {
  Entry ent[kMaxAlpha];
  uint32_t src[kMaxAlpha];
  idx_t size = 0;
};

// All finger lists of one point. Lists [0, n_complete) are complete (alpha
// entries); the rest is the tail, with strictly decreasing sizes ending in
// the empty list of radius 0. List 0 may itself be a tail list (base_size <
// alpha) when s_i has fewer than alpha successors.
struct FingerLists {
  idx_t alpha = 0;
  idx_t stride = 0;      // lists 0, stride, 2 stride, ... are stored in full
  bool has_adv = false;  // advance pointers are maintained (Alg. 4/5)

  FingerLists() = default;
  explicit FingerLists(idx_t a, bool with_adv = false, idx_t checkpoint_stride = 0)
      : alpha(a), stride(checkpoint_stride ? checkpoint_stride : default_stride(a)), has_adv(with_adv) {
    assert(a >= 1 && a <= kMaxAlpha && stride >= 1 && stride <= max_stride(a));
  }
  // A delta names the slot it kills in one byte: alpha + stride - 1 <= 256.
  static idx_t max_stride(idx_t a) { return 257 - a; }
  // A copy owns its buffer, whether or not the original did.
  FingerLists(const FingerLists& o)
      : alpha(o.alpha), stride(o.stride), has_adv(o.has_adv), n_complete_(o.n_complete_), n_lists_(o.n_lists_),
        base_size_(o.base_size_) {
    if (o.cap_ > 0) {
      allocate(o.cap_);
      std::memcpy(buf_, o.buf_, layout(cap_).bytes);
    }
  }
  FingerLists(FingerLists&& o) noexcept { swap(o); }
  FingerLists& operator=(FingerLists o) noexcept {
    swap(o);
    return *this;
  }
  ~FingerLists() { release(); }
  void swap(FingerLists& o) noexcept {
    std::swap(alpha, o.alpha);
    std::swap(stride, o.stride);
    std::swap(has_adv, o.has_adv);
    std::swap(n_complete_, o.n_complete_);
    std::swap(n_lists_, o.n_lists_);
    std::swap(cap_, o.cap_);
    std::swap(ckpt_off_, o.ckpt_off_);
    std::swap(adv_off_, o.adv_off_);
    std::swap(tail_off_, o.tail_off_);
    std::swap(base_size_, o.base_size_);
    std::swap(owned_, o.owned_);
    std::swap(buf_, o.buf_);
  }

  // Lists between full copies: max(alpha, 8). A block's evictors then fill
  // at least a cache line, the copies add at most 12 bytes per list with
  // pointers (8 without) to the 17 (13) of the deltas and radii, and every
  // pointer hint is at most stride - 1 lists old (see AdvanceNav). Measured
  // on uniform 2D (docs/PLAN.md section 9): halving the stride costs 20-25 %
  // more memory for no measurable query time, doubling it saves 10-20 %
  // memory for 5-10 % slower queries.
  static idx_t default_stride(idx_t a) { return std::min(std::max<idx_t>(a, 8), max_stride(a)); }

  idx_t num_lists() const { return n_lists_; }
  idx_t num_complete() const { return n_complete_; }
  bool is_complete(idx_t k) const { return k < n_complete_; }
  // Tail list k has num_lists() - 1 - k entries: the last one is empty and
  // every tail list is its predecessor minus one entry.
  idx_t size(idx_t k) const {
    assert(k < n_lists_);
    return k < n_complete_ ? alpha : n_lists_ - 1 - k;
  }
  // Lists stored as checkpoints and deltas: the complete lists, or list 0
  // alone when it is a tail list.
  idx_t stored() const { return n_complete_ > 0 ? n_complete_ : std::min<idx_t>(n_lists_, 1); }
  idx_t base_size() const { return base_size_; }
  bool is_checkpoint(idx_t k) const { return k % stride == 0; }
  static idx_t blocks(idx_t stored_lists, idx_t stride) { return stored_lists == 0 ? 0 : (stored_lists - 1) / stride + 1; }

  dist_t radius(idx_t k) const {
    assert(k < n_lists_);
    return radius_ptr()[k];
  }
  dist_t last_radius() const { return radius(n_lists_ - 1); }
  const dist_t* radius_data() const { return radius_ptr(); }
  // The delta that turns list k-1 into complete list k (1 <= k < num_complete):
  // the entry appended, and the slot removed, numbered within the block of
  // list k-1 (0..alpha-1 its checkpoint entries, alpha + t its t-th evictor).
  const Entry& evictor(idx_t k) const {
    assert(k >= 1 && k < n_complete_);
    return evictor_ptr()[k - 1];
  }
  idx_t evicted(idx_t k) const {
    assert(k >= 1 && k < n_complete_);
    return evicted_ptr()[k - 1];
  }
  // The first pointer slot of the block holding list k (slot tags of a
  // materialized list are this plus the slot within the block).
  size_t block_base(idx_t k) const { return static_cast<size_t>(k / stride) * block_slots(); }
  // Checkpoint list b * stride in full (block b; zero-padded for list 0 of a
  // provisional point).
  const Entry* checkpoint(idx_t b) const {
    assert(b < blocks(stored(), stride));
    return ckpt_ptr() + static_cast<size_t>(b) * alpha;
  }

  // Advance-pointer slots. Block b (lists b*stride .. b*stride + stride - 1)
  // holds alpha slots for its checkpoint list followed by one slot, the
  // evictor's, for each later list; slots are therefore in creation order.
  // adv_at(adv_slot(k, e)) is the index of F_j(radius[k]) in F_j for j =
  // entry e of list k. Tail lists carry no pointers.
  size_t block_slots() const { return static_cast<size_t>(alpha) + stride - 1; }
  size_t adv_slots(idx_t stored_lists) const {
    return stored_lists + static_cast<size_t>(blocks(stored_lists, stride)) * (alpha - 1);
  }
  bool has_adv_slot(idx_t k, idx_t e) const { return is_checkpoint(k) || e + 1 == alpha; }
  size_t adv_slot(idx_t k, idx_t e) const {
    assert(has_adv_slot(k, e) && e < alpha);
    const idx_t b = k / stride, q = k - b * stride;
    return b * block_slots() + (q == 0 ? e : alpha + q - 1);
  }
  idx_t adv_at(size_t s) const {
    assert(has_adv && s < adv_slots(stored()));
    return adv_ptr()[s];
  }
  idx_t& adv_at(size_t s) {
    assert(has_adv && s < adv_slots(stored()));
    return adv_ptr()[s];
  }
  // Inverse of adv_slot: (list, position) of a slot.
  std::pair<idx_t, idx_t> slot_pos(size_t s) const {
    const size_t b = s / block_slots(), rem = s - b * block_slots();
    if (rem < alpha) return {static_cast<idx_t>(b * stride), static_cast<idx_t>(rem)};
    return {static_cast<idx_t>(b * stride + rem - alpha + 1), alpha - 1};
  }
  const Entry& slot_entry(size_t s) const {
    const size_t b = s / block_slots(), rem = s - b * block_slots();
    return rem < alpha ? ckpt_ptr()[b * alpha + rem] : evictor_ptr()[b * stride + rem - alpha];
  }
  // For an entry of checkpoint list k that was kept from an earlier list
  // (k > 0, e < alpha - 1): the slot of the same point's most recent earlier
  // pointer, i.e. its evictor slot when it was appended within the block,
  // else its slot in the previous checkpoint. The push-down start (Alg. 5).
  size_t prev_slot(idx_t k, idx_t e) const {
    assert(k > 0 && is_checkpoint(k) && e + 1 < alpha);
    const idx_t j = checkpoint(k / stride)[e].idx;
    const Entry* ev = evictor_ptr();
    for (idx_t t = k - 1; t + stride > k; t--)
      if (ev[t - 1].idx == j) return adv_slot(t, alpha - 1);
    const Entry* prev = checkpoint(k / stride - 1);
    for (idx_t p = 0; p < alpha; p++)
      if (prev[p].idx == j) return adv_slot(k - stride, p);
    assert(false && "checkpoint entry missing from the previous checkpoint and the block's evictors");
    return 0;
  }

  // Materializes list k into L: entries in idx order and, with WithSrc, each
  // entry's pointer slot. A stored list is the live slots of its block: the
  // checkpoint's entries and then the block's evictors (in idx order, since
  // every evictor has the largest idx so far); replaying the at most
  // stride - 1 deltas is two bit operations each on the live mask, then one
  // gather, O(alpha + stride). A tail list, and the last stored list once
  // the tail is built, is filtered from the tail base by the rank at which
  // each entry drops out, O(alpha).
  template <bool WithSrc = true>
  void materialize(idx_t k, ListBuf& L) const {
    assert(k < n_lists_);
    const idx_t last = stored() - 1;
    if (k >= last && n_lists_ > last + 1) {  // tail built: list `last` and the tail are filtered from it
      const idx_t r = k - last;  // entries of rank < r have dropped out
      const Entry* ent = tail_ent_ptr();
      const idx_t* src = tail_src_ptr();
      const uint8_t* rank = tail_rank_ptr();
      const idx_t sz = last == 0 ? base_size_ : alpha;
      idx_t w = 0;
      for (idx_t e = 0; e < sz; e++) {
        if (rank[e] < r) continue;
        L.ent[w] = ent[e];
        if (WithSrc) L.src[w] = src[e];
        w++;
      }
      L.size = w;
      return;
    }
    const idx_t b = k / stride, c = b * stride, m = k - c;
    const idx_t sz = k == 0 ? base_size_ : alpha;
    const Entry* ck = checkpoint(b);
    const Entry* ev = evictor_ptr() + c;
    const uint8_t* dead = evicted_ptr() + c;
    const size_t base = b * block_slots();
    idx_t w = 0;
    if (block_slots() <= 64) {
      uint64_t live = sz >= 64 ? ~uint64_t{0} : (uint64_t{1} << sz) - 1;
      for (idx_t t = 0; t < m; t++) live = (live & ~(uint64_t{1} << dead[t])) | (uint64_t{1} << (alpha + t));
      // checkpoint entries first, then the block's evictors: both in idx order
      for (uint64_t x = alpha >= 64 ? live : live & ((uint64_t{1} << alpha) - 1); x; x &= x - 1) {
        const idx_t s = static_cast<idx_t>(__builtin_ctzll(x));
        L.ent[w] = ck[s];
        if (WithSrc) L.src[w] = static_cast<uint32_t>(base + s);
        w++;
      }
      for (uint64_t x = alpha >= 64 ? 0 : live >> alpha; x; x &= x - 1) {
        const idx_t t = static_cast<idx_t>(__builtin_ctzll(x));
        L.ent[w] = ev[t];
        if (WithSrc) L.src[w] = static_cast<uint32_t>(base + alpha + t);
        w++;
      }
    } else {  // up to 256 slots
      uint64_t live[4] = {};
      for (idx_t s = 0; s < sz; s++) live[s / 64] |= uint64_t{1} << (s % 64);
      for (idx_t t = 0; t < m; t++) {
        live[dead[t] / 64] &= ~(uint64_t{1} << (dead[t] % 64));
        live[(alpha + t) / 64] |= uint64_t{1} << ((alpha + t) % 64);
      }
      for (idx_t word = 0; word < 4; word++)
        for (uint64_t x = live[word]; x; x &= x - 1) {
          const idx_t s = static_cast<idx_t>(word * 64 + __builtin_ctzll(x));
          L.ent[w] = s < alpha ? ck[s] : ev[s - alpha];
          if (WithSrc) L.src[w] = static_cast<uint32_t>(base + s);
          w++;
        }
    }
    assert(w == sz);
    L.size = sz;
  }

  // F_i(r): index of the first list with radius <= r (the paper's Table 2
  // and Alg. 2; its Sec. 3.3 text says "last", which is a slip), by binary
  // search over the non-increasing radius array. The last list is empty with
  // radius 0, so the result is valid for every r >= 0.
  idx_t locate(dist_t r) const {
    assert(n_lists_ > 0 && r >= 0);
    const dist_t* rad = radius_ptr();
    auto it = std::partition_point(rad, rad + n_lists_, [r](dist_t x) { return x > r; });
    return static_cast<idx_t>(it - rad);
  }

  // Same result as locate(r), reached by walking up or down from list k
  // (align in Alg. 4). Cheap when k is already close.
  idx_t align(idx_t k, dist_t r) const {
    assert(k < n_lists_ && r >= 0);
    const dist_t* rad = radius_ptr();
    while (k > 0 && rad[k - 1] <= r) k--;
    while (rad[k] > r) k++;
    return k;
  }

  // Appends list 0: sz <= alpha entries sorted by idx ascending. Its pointer
  // slots, if any, start at 0.
  void push_first(const Entry* first, idx_t sz) {
    assert(n_lists_ == 0 && sz <= alpha);
    ensure(1);
    radius_ptr()[0] = max_dist(first, sz);
    base_size_ = static_cast<uint8_t>(sz);
    Entry* ck = ckpt_ptr();
    for (idx_t e = 0; e < sz; e++) ck[e] = first[e];
    for (idx_t e = sz; e < alpha; e++) ck[e] = Entry{0, 0};
    if (has_adv) std::fill_n(adv_ptr(), alpha, idx_t{0});
    n_lists_ = 1;
    if (sz == alpha) n_complete_ = 1;
  }

  // Appends the complete list that is `last` (the last list, complete and
  // not followed by a tail, materialized with its slot tags) with position f
  // removed and ev appended, which must have a larger idx than every other
  // entry; then turns `last` into that list. Its pointer slots, if any, start
  // at 0.
  void push_evict(ListBuf& last, idx_t f, Entry ev) {
    assert(n_complete_ > 0 && n_complete_ == n_lists_ && f < alpha && last.size == alpha);
    assert(ev.idx > last.ent[alpha - 1].idx);
    const idx_t k = n_complete_;
    const size_t base = block_base(k - 1);
    assert(last.src[f] >= base && last.src[f] - base < block_slots());
#ifndef NDEBUG
    {
      ListBuf prev;
      materialize<true>(k - 1, prev);
      assert(std::equal(prev.ent, prev.ent + alpha, last.ent) && std::equal(prev.src, prev.src + alpha, last.src) &&
             "push_evict: `last` is not the last list");
    }
#endif
    ensure(k + 1);
    evictor_ptr()[k - 1] = ev;
    evicted_ptr()[k - 1] = static_cast<uint8_t>(last.src[f] - base);
    dist_t rad = ev.dist;
    for (idx_t e = f; e + 1 < alpha; e++) {
      last.ent[e] = last.ent[e + 1];
      last.src[e] = last.src[e + 1];
    }
    last.ent[alpha - 1] = ev;
    for (idx_t e = 0; e + 1 < alpha; e++) rad = std::max(rad, last.ent[e].dist);
    radius_ptr()[k] = rad;
    if (is_checkpoint(k)) {
      Entry* ck = ckpt_ptr() + static_cast<size_t>(k / stride) * alpha;
      for (idx_t e = 0; e < alpha; e++) {
        ck[e] = last.ent[e];
        last.src[e] = static_cast<uint32_t>(adv_slot(k, e));
      }
      if (has_adv) std::fill_n(adv_ptr() + adv_slot(k, 0), alpha, idx_t{0});
    } else {
      last.src[alpha - 1] = static_cast<uint32_t>(adv_slot(k, alpha - 1));
      if (has_adv) adv_ptr()[adv_slot(k, alpha - 1)] = 0;
    }
    n_lists_ = n_complete_ = k + 1;
  }

  // Drops every tail list (size < alpha). A point without a complete list
  // is left with no lists at all.
  void truncate_tail() {
    if (n_complete_ == 0) clear();
    else n_lists_ = n_complete_;
  }

  // From the last list, repeatedly removes the farthest entry (tie rule) and
  // appends the result, until the empty list has been appended: the last
  // list is kept as the tail base with each entry's removal rank, and the
  // radii are stored. With no lists at all, appends just the empty list.
  void build_tail() {
    if (n_lists_ == 0) {
      push_first(nullptr, 0);
      return;
    }
    assert(n_lists_ == stored() && "tail already built");
    const idx_t last = n_lists_ - 1;
    ListBuf L;
    materialize<true>(last, L);
    std::copy_n(L.ent, L.size, tail_ent_ptr());
    std::copy_n(L.src, L.size, tail_src_ptr());
    // Removing the r farthest entries leaves radius d[r] with the distances
    // sorted descending (equal distances give equal radii, so the tie rule
    // only matters for the ranks).
    uint8_t order[kMaxAlpha];
    for (idx_t e = 0; e < L.size; e++) order[e] = static_cast<uint8_t>(e);
    std::sort(order, order + L.size, [&L](uint8_t a, uint8_t b) { return farther(L.ent[a], L.ent[b]); });
    uint8_t* rank = tail_rank_ptr();
    dist_t* rad = radius_ptr() + n_lists_;
    for (idx_t r = 0; r < L.size; r++) {
      rank[order[r]] = static_cast<uint8_t>(r);
      if (r > 0) *rad++ = L.ent[order[r]].dist;
    }
    if (L.size > 0) *rad++ = 0;
    n_lists_ = static_cast<idx_t>(rad - radius_ptr());
  }

  // Room for `lists` stored lists (the tail's radii are always included)
  // before the buffer has to grow.
  void reserve(size_t lists) {
    const idx_t want = static_cast<idx_t>(std::min<size_t>(std::max<size_t>(lists, 1), idx_t{1} << 30));
    if (want > cap_) reallocate(want);
  }
  // Bytes a buffer for `lists` stored lists needs (see layout).
  size_t bytes_for(idx_t lists) const { return layout(lists).bytes; }
  // Uses `buf`, of at least bytes_for(lists) bytes owned by the caller and
  // outliving this object or its next attach/reallocation, as the buffer.
  // Drops the current contents.
  void attach(std::byte* buf, idx_t lists) {
    release();
    const Layout L = layout(lists);
    buf_ = buf;
    cap_ = lists;
    owned_ = false;
    ckpt_off_ = static_cast<uint32_t>(L.ckpt);
    adv_off_ = static_cast<uint32_t>(L.adv);
    tail_off_ = static_cast<uint32_t>(L.tail);
    clear();
  }

  // Empties the lists but keeps the buffer.
  void clear() {
    n_lists_ = 0;
    n_complete_ = 0;
    base_size_ = 0;
  }

  // Bytes of the stored representation (radii, deltas, checkpoints, pointers,
  // tail base) and of the allocation holding it.
  size_t logical_bytes() const {
    const idx_t s = stored();
    if (s == 0) return 0;
    return sizeof(dist_t) * n_lists_ + (sizeof(Entry) + 1) * (s - 1) + sizeof(Entry) * blocks(s, stride) * alpha +
           (has_adv ? sizeof(idx_t) * adv_slots(s) : 0) + (n_lists_ > s ? (sizeof(Entry) + sizeof(idx_t) + 1) * alpha : 0);
  }
  size_t bytes() const { return cap_ == 0 ? 0 : layout(cap_).bytes; }
  idx_t capacity() const { return cap_; }
  bool owns_buffer() const { return owned_; }  // outgrew its slab slice (or is a copy)

  // Structural invariants checkable without the metric: the complete/tail
  // split and the tail's shape, non-increasing radii equal to the max cached
  // distance, idx sorted and > owner, every evicted position the farthest
  // entry of its predecessor, every evictor closer than the old radius and
  // of lower priority than the rest, and every list as materialized (through
  // the checkpoints and the tail base) equal to a replay of all deltas from
  // list 0. Returns "" when they hold.
  std::string validate(idx_t owner) const {
    auto fail = [&](const std::string& what, idx_t k) {
      return "F_" + std::to_string(owner) + " list " + std::to_string(k) + ": " + what;
    };
    if (n_lists_ == 0) return fail("no lists", 0);
    if (stride < 1 || alpha < 1 || alpha > kMaxAlpha) return fail("alpha or stride", 0);
    if (n_complete_ > 0 ? (base_size_ != alpha || n_lists_ != n_complete_ + alpha) : n_lists_ != base_size_ + 1u)
      return fail("tail shape", 0);
    if (cap_ < stored()) return fail("capacity", 0);
    ListBuf R, L;  // R: replayed delta by delta from list 0; L: as materialized
    for (idx_t k = 0; k < n_lists_; k++) {
      if (k == 0) {
        std::copy_n(checkpoint(0), base_size_, R.ent);
        for (idx_t e = 0; e < base_size_; e++) R.src[e] = e;
        R.size = base_size_;
      } else if (k < n_complete_) {
        const Entry ev = evictor(k);
        const size_t dead = block_base(k - 1) + evicted(k);
        idx_t f = 0;
        while (f < alpha && R.src[f] != dead) f++;
        if (f == alpha) return fail("evicted slot not in the previous list", k);
        if (f != farthest(R.ent, alpha)) return fail("evicted entry is not the farthest", k);
        if (!(ev.dist < radius(k - 1))) return fail("evictor not closer than old radius", k);
        if (!(ev.idx > R.ent[alpha - 1].idx)) return fail("evictor not of lowest priority", k);
        std::copy(R.ent + f + 1, R.ent + alpha, R.ent + f);
        std::copy(R.src + f + 1, R.src + alpha, R.src + f);
        R.ent[alpha - 1] = ev;
        R.src[alpha - 1] = static_cast<uint32_t>(adv_slot(k, alpha - 1));
        if (is_checkpoint(k)) {
          if (!std::equal(R.ent, R.ent + alpha, checkpoint(k / stride))) return fail("checkpoint differs from the replay", k);
          for (idx_t e = 0; e < alpha; e++) R.src[e] = static_cast<uint32_t>(adv_slot(k, e));
        }
      } else {
        const idx_t f = farthest(R.ent, R.size);
        std::copy(R.ent + f + 1, R.ent + R.size, R.ent + f);
        std::copy(R.src + f + 1, R.src + R.size, R.src + f);
        R.size--;
      }
      materialize<true>(k, L);
      if (L.size != R.size || !std::equal(L.ent, L.ent + L.size, R.ent)) return fail("materialized list differs from the replay", k);
      if (!std::equal(L.src, L.src + L.size, R.src)) return fail("slot tags differ from the replay", k);
      if (L.size != size(k)) return fail("size", k);
      dist_t rad = 0;
      for (idx_t e = 0; e < L.size; e++) {
        const Entry& x = L.ent[e];
        if (x.idx <= owner) return fail("idx <= owner", k);
        if (e > 0 && !(L.ent[e - 1].idx < x.idx)) return fail("not sorted by idx", k);
        if (x.dist < 0) return fail("negative dist", k);
        rad = std::max(rad, x.dist);
      }
      if (radius(k) != rad) return fail("radius != max dist", k);
      if (k > 0 && radius(k) > radius(k - 1)) return fail("radius increases", k);
    }
    return "";
  }

 private:
  idx_t n_complete_ = 0;
  idx_t n_lists_ = 0;
  idx_t cap_ = 0;          // stored lists the buffer has room for
  uint32_t ckpt_off_ = 0;  // byte offsets of the sections for cap_ (see layout)
  uint32_t adv_off_ = 0;
  uint32_t tail_off_ = 0;
  uint8_t base_size_ = 0;
  bool owned_ = false;    // buf_ is ours to free (else a slice of the owner's slab)
  std::byte* buf_ = nullptr;

  // Byte offsets of the sections for a capacity of c stored lists. Radii
  // come first (c + alpha lists: the tail adds at most alpha), then the
  // evictors, the checkpoint blocks, the pointers, the tail base (entries,
  // slots), and the byte-sized evicted positions and tail ranks last.
  struct Layout {
    size_t evictor, ckpt, adv, tail, bytes;
  };
  Layout layout(idx_t c) const {
    Layout L;
    size_t o = sizeof(dist_t) * (static_cast<size_t>(c) + alpha);
    L.evictor = o;
    o += sizeof(Entry) * c;
    L.ckpt = o;
    o += sizeof(Entry) * static_cast<size_t>(blocks(c, stride)) * alpha;
    L.adv = o;
    if (has_adv) o += sizeof(idx_t) * adv_slots(c);
    L.tail = o;
    o += (sizeof(Entry) + sizeof(idx_t)) * alpha;  // tail entries and slot tags
    o += c + alpha;                                // evicted[c], tail_rank[alpha]
    L.bytes = (o + 7) & ~size_t{7};                // slab slices stay 8-byte aligned
    return L;
  }

  dist_t* radius_ptr() const { return reinterpret_cast<dist_t*>(buf_); }
  Entry* evictor_ptr() const { return reinterpret_cast<Entry*>(buf_ + sizeof(dist_t) * (static_cast<size_t>(cap_) + alpha)); }
  Entry* ckpt_ptr() const { return reinterpret_cast<Entry*>(buf_ + ckpt_off_); }
  idx_t* adv_ptr() const { return reinterpret_cast<idx_t*>(buf_ + adv_off_); }
  Entry* tail_ent_ptr() const { return reinterpret_cast<Entry*>(buf_ + tail_off_); }
  idx_t* tail_src_ptr() const { return reinterpret_cast<idx_t*>(buf_ + tail_off_ + sizeof(Entry) * alpha); }
  uint8_t* evicted_ptr() const {
    return reinterpret_cast<uint8_t*>(buf_ + tail_off_ + (sizeof(Entry) + sizeof(idx_t)) * alpha);
  }
  uint8_t* tail_rank_ptr() const { return evicted_ptr() + cap_; }

  void allocate(idx_t c) {
    const Layout L = layout(c);
    buf_ = parlay::allocator<std::byte>().allocate(L.bytes);
    owned_ = true;
    cap_ = c;
    ckpt_off_ = static_cast<uint32_t>(L.ckpt);
    adv_off_ = static_cast<uint32_t>(L.adv);
    tail_off_ = static_cast<uint32_t>(L.tail);
  }
  void release() {
    if (buf_ && owned_) parlay::allocator<std::byte>().deallocate(buf_, layout(cap_).bytes);
    buf_ = nullptr;
    owned_ = false;
    cap_ = 0;
  }
  // Moves the contents to an owned buffer for c >= stored() lists.
  void reallocate(idx_t c) {
    assert(c >= stored());
    FingerLists o(alpha, has_adv, stride);
    o.allocate(c);
    if (buf_) {
      const idx_t s = stored();
      std::copy_n(radius_ptr(), n_lists_, o.radius_ptr());
      if (s > 1) {
        std::copy_n(evictor_ptr(), s - 1, o.evictor_ptr());
        std::copy_n(evicted_ptr(), s - 1, o.evicted_ptr());
      }
      std::copy_n(ckpt_ptr(), static_cast<size_t>(blocks(s, stride)) * alpha, o.ckpt_ptr());
      if (has_adv) std::copy_n(adv_ptr(), adv_slots(s), o.adv_ptr());
      if (n_lists_ > s) {
        std::copy_n(tail_ent_ptr(), alpha, o.tail_ent_ptr());
        std::copy_n(tail_src_ptr(), alpha, o.tail_src_ptr());
        std::copy_n(tail_rank_ptr(), alpha, o.tail_rank_ptr());
      }
    }
    o.n_complete_ = n_complete_;
    o.n_lists_ = n_lists_;
    o.base_size_ = base_size_;
    swap(o);
  }
  // Room for `count` stored lists, doubling the buffer when it runs out.
  void ensure(idx_t count) {
    if (count <= cap_) return;
    reallocate(std::max<idx_t>(count, std::max<idx_t>(2 * cap_, 8)));
  }

  static dist_t max_dist(const Entry* first, idx_t sz) {
    dist_t rad = 0;
    for (idx_t e = 0; e < sz; e++) {
      assert(e == 0 || first[e - 1].idx < first[e].idx);
      rad = std::max(rad, first[e].dist);
    }
    return rad;
  }
};

// Bitwise comparison of the parts every builder must agree on: the radii and
// the entries of every list (whatever the checkpoint stride). with_adv also
// compares the stored advance pointers, which requires equal strides.
inline bool same_lists(const FingerLists& a, const FingerLists& b, bool with_adv = false) {
  if (a.alpha != b.alpha || a.num_lists() != b.num_lists() || a.num_complete() != b.num_complete()) return false;
  if (!std::equal(a.radius_data(), a.radius_data() + a.num_lists(), b.radius_data())) return false;
  ListBuf La, Lb;
  for (idx_t k = 0; k < a.num_lists(); k++) {
    a.materialize<false>(k, La);
    b.materialize<false>(k, Lb);
    if (La.size != Lb.size || !std::equal(La.ent, La.ent + La.size, Lb.ent)) return false;
  }
  if (with_adv) {
    if (a.has_adv != b.has_adv || a.stride != b.stride) return false;
    if (a.has_adv)
      for (size_t s = 0, n = a.adv_slots(a.stored()); s < n; s++)
        if (a.adv_at(s) != b.adv_at(s)) return false;
  }
  return true;
}

}  // namespace mskip
