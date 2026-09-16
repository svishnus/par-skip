# Implementation plan: parallel metric skip lists on ParlayLib

Paper: Ding, Garg, Gu, Sun. *Parallel Metric Skip Lists and Nearest Neighbor
Search.* SPAA '26 (`paper.pdf`). Full version with the Sec. 5.5 details
(Appendix B, Alg. 7): arXiv:2606.03129.

This document fixes the contracts (types, invariants, tie rules, function
signatures) so that pieces written by different people fit together and so that
the three builders — reference, sequential, parallel — produce **bit-identical**
structures, which is the main correctness test.

---

## 1. Ground rules

1. **0-indexed.** Points are stored in permuted order `pts[0..n)`; `pts[i]` is
   the paper's `s_{i+1}`. `perm[i]` = original index of `pts[i]`. Lower index
   = higher priority. All ranges `[l, r]` in the parallel build are inclusive.
2. **Only lower-priority points appear in a point's lists.** Every entry of
   `F_i` has `idx > i`.
3. **Strict comparisons as in the paper.** A point enters a list only if
   `d < radius`; the walk chooses `nxt` only if `d(nxt, q) < max(radius, d(cur, q))`.
   All comparisons are on *computed* distances; the structure is defined by
   them, not by exact arithmetic.
4. **Tie rule (must be identical in every builder):** "the farthest point" of a
   list is the entry with maximum `dist`, ties broken by the **largest `idx`**
   (lowest priority). Consequence: list `F_i[k]` is always the `alpha` smallest
   entries of its prefix under the order `(dist, idx)`, so the structure is a
   pure function of the permutation.
5. **Complete vs tail.** A list is *complete* iff it has exactly `alpha`
   entries. Complete lists form a prefix of `F_i`; everything after is the
   *tail* (sizes `t-1, t-2, ..., 0`, where `t` is the size of the last non-tail
   list; the last tail list is empty with radius 0). Tails are always rebuilt
   from the last complete list, so they may be discarded freely.
6. **Semantic invariant** (what the structure *means*; verified exactly, ties
   included):
   `F_i(r)` := first list of `F_i` with `radius <= r` =
   **the `alpha` highest-priority points among `{ j > i : d(s_i, s_j) <= r }`,
   or all of them if there are fewer than `alpha`.**
7. **`alpha` is a runtime parameter.** Exactness of all queries holds for any
   `alpha >= 1`; the paper's `16c^3` is only for the running-time analysis.
8. **Determinism.** Given the same permutation, `build_sequential()`,
   `build_parallel()` (any thread count, `PARLAY_SEQUENTIAL` or not) and the
   reference builder must produce identical `radius`, `size`, `entries`
   (and `adv`). Distances are always evaluated as `metric(owner or query,
   other)`, so a metric only has to be exactly symmetric if it is used that
   way.
9. **Rounding slack.** The exactness argument of the walk needs the triangle
   inequality on the computed distances, which float `L2` violates by an ulp
   now and then (a concrete triple is in `test_query.cpp`). Every metric
   declares `static constexpr dist_t slack` with
   `d(a, c) <= (d(a, b) + d(b, c)) * (1 + slack)` for computed values, and the
   walk multiplies every search radius by `1 + slack` (a larger ball keeps the
   argument valid; see section 3). A slack is relative, so it has to hold at
   every magnitude: the built-in metrics accumulate in double and round once
   to float (no subnormal squares), which makes `4 eps` enough with a factor
   of two to spare; exact metrics use 0.

---

## 2. Data model (`include/mskip/`)

```cpp
// types.hpp (exists)
using idx_t  = uint32_t;   // permutation position / list index
using dist_t = float;      // distances

// metric.hpp (exists)
// Metric concept: point_type; dist_t operator()(const point_type&, const point_type&) const
template <int D> using Point = std::array<dist_t, D>;
template <int D> struct L2 / L1 / Linf;

// finger_list.hpp
struct Entry { idx_t idx; dist_t dist; };       // dist = d(s_i, s_idx), cached

// All finger lists of one point. List k occupies entries[k*alpha, k*alpha + size[k]).
struct FingerLists {
  idx_t alpha;
  idx_t n_complete;                   // lists [0, n_complete) are complete; the rest is the tail
  bool  has_adv;                      // advance pointers are maintained (Phase 3/4)
  parlay::sequence<dist_t>  radius;   // radius[k] = max dist in list k, 0 if empty; non-increasing
  parlay::sequence<uint8_t> size;     // size[k] <= alpha; == alpha iff complete
  parlay::sequence<Entry>   entries;  // alpha-stride; each list sorted by idx ascending
  parlay::sequence<idx_t>   adv;      // alpha-stride, parallel to entries: adv[k*alpha+e] is the index
                                      // of F_j(radius[k]) in F_j, j = entries[k*alpha+e].idx.
                                      // Only complete lists carry pointers (see section 3).

  idx_t num_lists() const;
  idx_t num_complete() const;
  parlay::slice<const Entry*, const Entry*> list(idx_t k) const;
  idx_t locate(dist_t r) const;                     // F_i(r): first k with radius[k] <= r; binary search
  idx_t align(idx_t k, dist_t r) const;             // same result, walking up/down from k (Alg. 4)
  void  push_back(const Entry* first, idx_t sz);    // computes radius; asserts sorted-by-idx, sz <= alpha
  void  truncate_tail();                            // drop every list with size < alpha
  void  build_tail();                               // from the last list, repeatedly remove the farthest
                                                    // (tie rule) and push, until the empty list is pushed
  std::string validate(idx_t owner) const;          // structural invariants, "" if they hold
};
```

Why `alpha`-stride with explicit sizes: `F_i[k]` is O(1)-addressable, `radius`
is a contiguous array for binary search, and tail lists need no padding logic.
Memory is inherently Θ(α² ln n) entries per point (α entries × α ln n lists);
fine up to n ≈ 10⁵ at α = 8, or 10⁶ at α ≤ 4. Compaction is a Phase 5 concern.

```cpp
// metric_skip_list.hpp (to write)
template <class Metric>
class MetricSkipList {
 public:
  using point_type = typename Metric::point_type;
  // Shuffles pts with `seed` (parlay::random_permutation) and stores perm.
  MetricSkipList(parlay::sequence<point_type> pts, idx_t alpha, Metric m = {}, uint64_t seed = 0);

  void build_sequential(bool advance = true);                       // Alg. 5 (advance) or Alg. 2
  void build_parallel(size_t seq_base = 1024, bool advance = true); // Alg. 6 + Sec. 5.5 (advance)
  bool has_advance() const;                                         // queries then use Alg. 4

  idx_t                    nearest(const point_type& q) const;                 // original index
  parlay::sequence<idx_t>  knn(const point_type& q, idx_t k) const;            // sorted by distance
  parlay::sequence<idx_t>  range(const point_type& q, dist_t delta) const;     // open ball d < delta

  // accessors for tests/bench
  idx_t n() const; idx_t alpha() const;
  const point_type& point(idx_t i) const;            // s_i
  dist_t dist(idx_t i, idx_t j) const;               // metric(point(i), point(j))
  const parlay::sequence<idx_t>& permutation() const;
  const FingerLists& lists(idx_t i) const;
  const parlay::sequence<idx_t>& control() const;      // C[i]: where the walk that built F_i stopped
  const parlay::sequence<idx_t>& control_list() const; // K[i]: index of the focus list in F_{C[i]} there
  const BuildStats& stats() const;                     // walks, steps, align moves, merges, forest depth
};
```

`control()` is recorded by every builder: the parallel build resumes each walk
exactly where it was blocked, so its final `C[i]`, `K[i]` equal the sequential
walk's stop point (tested).

---

## 3. The random walk (`walk.hpp`) — one routine for construction and all queries

State: focus point `cur`, a candidate set `K` (policy object), a target `q`.

```cpp
// Policy concept (one per use):
//   dist_t radius() const;                        // radius of the candidate set
//   void   offer(idx_t j, dist_t dj, idx_t hint); // called with the chosen nxt; inserts iff dj < radius();
//                                                 // hint = start index in F_j for its advance pointer
//   void   stop(idx_t cur, idx_t k);              // called once when the walk terminates at F_cur[k]
// Nav concept (how F* is found):
//   idx_t focus(const FingerLists& F, dist_t r);        // index of F(r)
//   idx_t hint(const FingerLists& F, idx_t k, idx_t e); // start index in F_j for entry e of list k
//   void  hop(idx_t h);                                 // the walk moves to that entry's point
template <class Metric, class Policy, class Nav>
size_t random_walk(const MetricSkipList<Metric>& S, const point_type& q, Policy& K, idx_t cur, Nav& nav);
```

Two navigators give the same `F*`, hence identical structures and answers:
`BinarySearchNav` (`locate`, Alg. 2/3) and `AdvanceNav` (Alg. 4/5: keeps the
index `k` of the focus list, `hop` sets it to the chosen entry's advance
pointer, `focus` corrects it with `align`). Tail lists carry no pointers
(as in the paper); their entries are a subset of the last complete list,
whose pointer is used instead (0 when the point has no complete list), then
`align` corrects it. No binary search is ever needed in advance mode.

```
random_walk(q, K, cur):
  loop:
    dc  = d(s_cur, q)
    r   = (dc + max(K.radius(), dc)) * (1 + slack)   // ball around cur that must contain any improvement
    L   = F_cur.list(F_cur.locate(r))       // F* in the paper
    thr = max(K.radius(), dc)
    nxt = none
    for e in L in ascending idx:            // "first j with the property" = highest priority
      dj = d(s_{e.idx}, q)
      if dj < thr: nxt = e.idx; K.offer(e.idx, dj); break
    if nxt == none:
      if |L| < alpha: K.stop(cur); return   // F* is a tail list: nothing to the right can improve K
      nxt = L.back().idx                    // lowest-priority entry: walk "outward"
    cur = nxt
```

Invariant maintained: `K` is exact for the prefix `{.., cur}` (for construction:
for `{i+1, .., cur}`), and `nxt` is chosen so that no point strictly between
`cur` and `nxt` in priority order could improve `K`. This is what makes every
query exact for any `alpha`. The argument only uses that `F*` is the `alpha`
highest-priority points of *some superset* of `B(cur, dc + thr)` among the
points after `cur`, so a larger radius (the slack) is harmless. Every hop goes
to an entry with a larger index than `cur`, so a walk ends after at most `n`
iterations whatever the metric does (NaN distances give garbage, not a hang).

Policies:

| use          | initial K                         | initial cur | `radius()`           | `offer(j, dj)`                                   |
|--------------|-----------------------------------|-------------|----------------------|--------------------------------------------------|
| build `F_i`  | `F_i[0] = {i+1 .. i+alpha}`       | `i+alpha`   | last list's radius   | if `dj < radius`: new list = last list with farthest replaced by `(j,dj)`, kept sorted by idx; `push_back` |
| nearest      | `{0}`                             | `0`         | `d(s_m, q)`          | if `dj < radius`: `m = j`                         |
| knn(k)       | `{0 .. k-1}` (return all if n ≤ k) | `k-1`       | max dist in K        | if `dj < radius`: replace farthest (tie rule)     |
| range(δ)     | ∅ (check `s_0` explicitly)        | `0`         | `δ` (constant)       | report `j`                                        |

Construction's `stop(cur, k)` builds the tail and records `C[i] = cur`, `K[i] = k`
(used by the parallel build to resume). Special case: if `i + alpha >= n`,
`F_i[0]` is short; it is already exact, so build the tail and skip the walk.

Advance pointers (Alg. 5, `build_sequential(true)`): the new list's entries
kept from the previous list start from that list's pointer ("push-down"),
the evictor starts from `hint` = the focus list's pointer for it ("shift
focus"); each is then `align`ed in `F_j` to the new radius. The first list
starts every pointer at `F_j[0]`. Measured on uniform 2D, α = 4: locating
`F*` costs ≈ 2.4 align moves per walk step against ≈ 5–6 comparisons for the
binary search; pointer maintenance costs ≈ 4 moves per step (α pointers per
new list). Sequential build time is unchanged; nearest-neighbor queries are
≈ 10 % faster.

---

## 4. Modules and who could write them

| # | File | Contents | Depends on |
|---|------|----------|------------|
| A | `finger_list.hpp` | `Entry`, `FingerLists` (§2), `replace_farthest`, `build_tail`, debug `check_invariants()` | types |
| B | `metric_skip_list.hpp` | class shell, shuffling, accessors | A, metric |
| C | `walk.hpp` | `random_walk` + the four policies | B |
| D | `build_seq.hpp` | `build_sequential()`: `for i = n-1 downto 0: build_finger_lists(i)` | C |
| E | `query.hpp` | `nearest`, `knn`, `range` | C |
| F | `reference.hpp` | definition-literal builder, `O(n·alpha)` per point (§5) | A, metric |
| G | `build_par.hpp` | `build_parallel()`: D&C + control tree + resume (§6) | C, D |
| H | `data.hpp` | generators: uniform `[0,1]^D`, Gaussian clusters, duplicates, collinear; seeded | metric |
| I | `tests/*.cpp` | §7 | all |
| J | `bench/*.cpp` | §8 | all |
| K | `advance` pointers | in C (`AdvanceNav`), D (`BuildPolicy::settle`) and G (`fixup`): Phase 3 (Alg. 4/5) and Phase 4 (section 6.1 below) | C, D, G |

A, C, D, E, G are the algorithmic core; F, H, I, J are plumbing. All modules
exist; `metric_skip_list.hpp` includes C, D, E, G at its end, so users include
only it (plus `reference.hpp` / `data.hpp` for tests).

---

## 5. Reference builder (`reference.hpp`) — the definition, literally

```
reference_lists(S, i):
  L = entries for {i+1 .. min(i+alpha, n-1)}, sorted by idx;  F = [L]
  if |L| == alpha:                              // otherwise L is already exact
    for j = i+alpha+1 .. n-1:                   // permutation order; i+alpha is in L
      if d(i, j) < radius(L): L = replace_farthest(L, (j, d)); F.push_back(L)
  build_tail(F)
```
O(n·alpha) per point → O(n²·alpha) total; used on n ≤ ~2000.

Semantic checker: for random `r` (and `r` = each radius, and `r` just below
each radius): `F_i.list(F_i.locate(r))` == the `alpha` smallest-index points in
`{ j > i : d(i,j) <= r }` (brute force). This is exact even with ties (rule 4).

---

## 6. Parallel build (`build_par.hpp`, Alg. 6 with the gaps filled)

```
build_parallel():  C[i] = i for all i;  parallel_build(0, n-1)

parallel_build(l, r):
  if r - l + 1 <= max(alpha, seq_base):          // base case (paper: r - l + 1 <= alpha)
    for i = r downto l: build_point(i, r)        // = sequential build of the range; provisional if i+alpha > r
    return
  m = (l + r) / 2
  par_do( parallel_build(l, m), parallel_build(m+1, r) )
  merge(l, m, r)

merge(l, m, r):                                   // complete the left half against [m+1..r]
  children = group_by_index( {(C[i]-l, i) : i in [l..m], C[i] != i}, m-l+1 )
  frontier = { i in [l..m] : C[i] == i }          // roots of the control forest
  while frontier not empty:
    parallel_for i in frontier: resume(i, r)
    frontier = flatten( children[f-l] for f in frontier )

resume(i, r):
  F_i.truncate_tail()                             // keep complete lists only
  if C[i] == i:                                   // never had a complete list
    if i + alpha > r: provisional(i, r); return   // still fewer than alpha successors
    F_i = [ {i+1 .. i+alpha} ];  cur = i + alpha; k = 0
  else:
    cur = C[i]; k = K[i]                          // resume where the walk was blocked
  random_walk(s_i, BuildPolicy(F_i), cur, nav(k)) // stop() sets C[i], K[i] and builds the tail

provisional(i, r):  F_i = tail-only lists over {i+1 .. r}   (fewer than alpha points)
```

Why this is sound (worth keeping in mind while coding):

* `C[i] > i` always (walks start at `i+alpha`), and `C[i] ∈ [l..m]` when `i`
  is in the left half, so the control structure is a forest rooted at
  `C[i] == i` nodes; height O(log n) whp (Lemma 5.1).
* For `i ∈ [l..m]`, the last merge that processed `i` (the shallowest level
  inside `[l..m]` at which `i` was in a *left* half; descending through right
  halves keeps the right endpoint) had right endpoint exactly `m`, so `F_i`
  is complete w.r.t. `m` and its walk was blocked at a tail list `T` of
  `F_{C[i]}` w.r.t. `m`. On resume the focus list is `F_{C[i]}(r)` w.r.t. `r`
  for the same `r`; structurally, the complete lists of `F_x` w.r.t. `m` are a
  prefix of those w.r.t. `r`, and `F_x(r)` w.r.t. `r` starts with the entries
  of `T` (only entries farther than `r` are ever evicted before the radius
  drops to `≤ r`) followed by points of `(m, r]`. `T`'s entries have no
  property, so the first hop lands in `(m, r]` — a structural fact that needs
  no metric property, which is why par == seq holds even where rounding
  breaks the triangle inequality — and every later focus point is in the
  right half, which is already complete. **The only left-half list a resumed
  walk reads is `F_{C[i]}`**, which finished in an earlier layer. Within a
  layer, resumes write disjoint `F_i` and read only finished lists — no
  synchronization beyond the layer barrier.
* `C[i] == i` ⟺ `F_i` has no complete list ⟺ `i + alpha > m`, so the roots of
  a merge's forest are exactly the last `min(alpha, m-l+1)` points of the left
  half (asserted in `merge`).
* Points with `i + alpha > r` never get complete lists at this level but can
  still be *focus points* of other walks, so they need the provisional tail
  (this is the gap in the paper's Alg. 6 line 9; at the top level it also
  gives the last `alpha` points the lists queries need).
* After the walk, the stop point becomes `C[i]` for the next level ("useful
  fact", Sec. 5.2), so control points are never recomputed.
* Paper's base case `C[l] <- l` is read as `C[i] <- i` for all `i` in the range.

* The sequential base case for ranges of ≤ `seq_base` points (default 1024)
  is the same computation as the recursion (a range built by `build_point`
  from `r` down to `l` leaves every `F_i` complete w.r.t. `r` and `C[i]` at
  the same stop point), it only skips the per-merge overhead. Tests run with
  `seq_base` = 0 (pure D&C down to `alpha`), 7 and the default.
* Any list index recorded earlier — `K[i]`, an advance pointer — is still a
  valid index whenever it is read: no `F_j` is read while it is rebuilt (a
  point rebuilt from provisional is a root of the merge's forest, roots read
  only right-half lists, and everything else reads it in a later layer or
  after the layers), and every rebuild ends with at least as many lists as
  before (a resume truncates only the tail and rebuilds one of the same
  length; a provisional point with `t + 1 ≤ alpha` lists gets `≥ alpha + 1`
  lists or `≥ t + 1` provisional ones). Asserted in `resume` and in `align`.

ParlayLib pieces: `parlay::par_do`, `parlay::parallel_for`,
`parlay::counting_sort_by_keys` (the control forest as CSR),
`parlay::flatten`/`filter`, `parlay::random_permutation`. Per-point
`FingerLists` are independent objects, so growing them from different workers
is safe.

Instrumentation (`BuildStats`): walks, steps, align moves (focus / pointers),
merges, control-forest layers and max depth. Observed on uniform 2D: forest
depth ≤ 9 for n = 10⁵ (mean 1.7 layers per merge), the parallel build does
5–15 % more walk steps than the sequential one (a resume re-examines the
blocking list), speedup 12.8× on 14 cores at n = 2·10⁵.

### 6.1 Advance pointers in the parallel build (Phase 4)

The paper's Sec. 5.5 / Appendix B (full version, Alg. 7) changes the walk
(radius `4·max(F.radius, d)`, a second phase that adds the last O(α) points
one by one, control point := last point added) and settles pointers through an
"advance tree". This implementation keeps the walk of section 6 (so the
structure, the control points and the walk length stay identical to the
sequential Alg. 5) and settles pointers as follows.

Invariant: every stored pointer is a valid index into its target's list array
whenever it is read (see the bullet on recorded indices above), and `align`
from any valid index is correct on the current lists. So a pointer is only
ever a *hint* while its target is not final; correctness needs only that
every pointer is `align`ed once more after its target's last change.

During a resumed walk of `i` in `merge(l, m, r)` (`BuildPolicy::settle`):

* target `j ∈ (m, r]` (finished for this level): `align` now. If the result
  is a complete list it is exact for good (complete lists are a stable prefix
  of `F_j`); if it lands in `F_j`'s tail and `r < n-1`, the slot is recorded
  as *pending* (`F_j` may still grow at a later level).
* target `j ∈ [l, m]`: `F_j` may be under construction by another worker in
  the same layer, so it is not read at all; the pointer keeps its start value
  and the slot is recorded as pending. Such pointers are never used as hints
  within the merge: every hop of a resumed walk lands in `(m, r]`.

After the last layer of the merge, `fixup(i)` runs in parallel over
`i ∈ [l, m]`: each pending slot is re-`align`ed (from the previous list's
pointer for the same target when there is one — the push-down; entry `e` of
list `k` is entry `e` or `e+1` of list `k-1` unless it is the evictor — so a
chain of deferred copies costs O(distance + length), not
O(distance × length)); slots that resolve to a complete list, or any slot when
`r == n-1`, leave the pending list. Pending slots stay in creation (= slot)
order, so the previous list's pointer has been fixed when a slot is reached.
After the top-level merge every pointer equals `F_j.locate(radius)` (tested:
bit-identical to the sequential Alg. 5 build over 1350 small configurations
and the large inputs of `test_build_par`; `BuildStats` counts deferred,
re-aligned and still-pending slots so the test can assert the paths ran).

Cost: a pending slot is touched once per level at which its owner is in a left
half, i.e. O(log n) times at most, each time for O(1) + moves; measured pointer
moves per walk step are 4.7 (parallel) vs 4.0 (sequential) on uniform 2D,
α = 4. Build time with pointers is ≈ 1.45× the binary-search parallel build
(the fix-up passes and 43 % more memory traffic); nearest-neighbor queries
then run ≈ 10 % faster. The walks never wait for a pointer, so the span
argument of Alg. 6 is unchanged.

---

## 7. Tests (`tests/`, no framework; `CHECK(cond)` macro, nonzero exit on failure)

| test | what |
|------|------|
| `test_reference` | structural invariants (radii non-increasing, sizes, sorted idx, `idx > i`, cached dists, tail shape) + semantic invariant (§5) on n ≤ 500, α ∈ {1,2,4,8}, L2/L1/Linf, D ∈ {1,2,8}, with duplicates and collinear inputs |
| `test_build_seq` | sequential == reference, exactly; n ∈ {1, 2, α, α+1, 50, 500, 2000} |
| `test_query` | nearest / knn / range == brute force on random queries, queries equal to data points, k ≥ n, δ = 0 |
| `test_build_par` | parallel == sequential, exactly; n up to 10⁵; also under `PARLAY_NUM_THREADS=1`, `SEQ=1`, and `DEBUG=1` (ASan/UBSan); records control-forest depth |
| `test_stats` | lists per point vs α·H_n, walk length vs log n (sanity, loose bounds) |
| `test_adversarial` | explicit (non-random) permutations: every point an evictor of `s_0`, sorted input without evictors, exponential gaps giving a control forest of depth n/α; all builders agree and queries stay exact |

Advance pointers are checked by definition (`check_advance`: every pointer of
every complete list equals `F_j.locate(radius)`, none pending) after the
sequential and the parallel build, in `test_build_seq` / `test_build_par`;
`test_query` runs every query in both navigation modes and holds the
rounding-slack regression (rule 9). `make test-all` runs the suite with the
default scheduler, `PARLAY_NUM_THREADS=1`, `SEQ=1` and `DEBUG=1`;
`make TSAN=1 test` runs it under ThreadSanitizer (ParlayLib's allocator
reports are suppressed in `tests/tsan.supp`).

The class also accepts an explicit permutation (`MetricSkipList(pts, alpha,
perm, metric)`) so tests can build adversarial orders; the default constructor
shuffles with `parlay::random_permutation(seed)`.

---

## 8. Benchmarks (`bench/`)

* construction: sequential vs parallel, threads 1..14, n = 10⁴..10⁶ (α small for large n)
* queries: batch of 10⁵ NN / kNN queries via `parallel_for`; throughput
* datasets: uniform 2D/3D/8D, Gaussian clusters (varying expansion rate)
* Phase 3/4: binary-search `locate` vs `advance`/`align`
* `bench/history.sh` re-measures every milestone commit with the same
  benchmark (worktree per milestone) into `bench/results/history.csv`;
  `bench/plot.py` draws `docs/plots/*.svg` from it (standard library only)

---

## 9. Phases and commits (conventional commits, signed)

| phase | status | scope | commits |
|-------|--------|-------|---------|
| 0 | done | scaffold, README, this plan | `chore: scaffold project with ParlayLib submodule and Makefile`, `docs: add README, implementation plan, and the SPAA '26 paper` |
| 1 | done | A B C D E F H, tests reference/seq/query | `feat(core): finger list storage`, `feat(core): random walk and sequential construction`, `feat(query): nearest, knn, range`, `test: reference builder and invariant checks` |
| 2 | done | G, test_build_par, bench | `feat(par): divide-and-conquer construction with control forest`, `test(par): parallel build matches sequential`, `bench: construction and query benchmarks` |
| 3 | done | advance/align, sequential (Alg. 4/5) | `feat(core): advance pointers and align` |
| 4 | done | advance/align, parallel (section 6.1; the arXiv full version's Alg. 7 was consulted) | `feat(par): maintain advance pointers in parallel build` |
| 5 | started | tuning: per-point reservation of the list arrays (done: resident set 3.3× → 1.9× logical, 1e6 build 2.0 → 1.2 s), α sweep, layout compaction, cache layout | `perf: reserve per-point list arrays for the expected list count`, `perf: ...` |

Two independent review rounds (after Phase 2 and after Phase 4) produced the
`fix(core)` rounding-slack and double-accumulation commits, the `fix(api)`
run-time checks, the Makefile `override`/`TSAN=1` change, the adversarial
and stress tests, and the `refactor(par)` of `fixup`.

Phase 5 notes from the measurements so far: the walk length is Θ(α ln n)
(every evictor is visited: ≈ 46 of 77 steps per point at n = 4·10⁵, α = 4),
and the time per step doubles from n = 2.5·10⁴ to 4·10⁵ because the
alpha-stride structure (≈ 1.5 KB per point at α = 4, ≈ 7 KB at α = 8 plus
pointers) is walked at random — compaction is the first thing to try. With
α ≤ 8 the walks are far longer than log n in 3D/8D (the analysis needs
α ≥ 16c³, c ≈ 2^D); queries stay exact, only the cost bound is lost.

---

## 10. Open decisions

* `range()` is an open ball (`d < delta`) to keep the walk's strict `<`; pass
  `nextafter(delta, inf)` for a closed ball.
* Resolved (Phase 3): tail lists get no `advance` pointers (paper); when `F*`
  is a tail list, the pointer of the last complete list — which contains every
  tail entry — is the start, and `align` corrects it. No binary search.
* Resolved (Phase 4): pointers are settled by per-point pending lists and a
  fix-up pass after each merge (section 6.1) instead of the paper's advance
  tree; the walk is unchanged from Alg. 6.
* Memory compaction (slot-history representation) only if needed in Phase 5.
