# Implementation plan: parallel metric skip lists on ParlayLib

Paper: Ding, Garg, Gu, Sun. *Parallel Metric Skip Lists and Nearest Neighbor
Search.* SPAA '26 (`paper.pdf`). Full version with the Sec. 5.5 details:
arXiv:2606.03129.

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
   reference builder must produce identical `radius`, `size`, `entries`.

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

// finger_list.hpp (to write)
struct Entry { idx_t idx; dist_t dist; };       // dist = d(s_i, s_idx), cached

// All finger lists of one point. List k occupies entries[k*alpha, k*alpha + size[k]).
struct FingerLists {
  idx_t alpha;
  parlay::sequence<dist_t>  radius;   // radius[k] = max dist in list k, 0 if empty; non-increasing
  parlay::sequence<uint8_t> size;     // size[k] <= alpha; == alpha iff complete
  parlay::sequence<Entry>   entries;  // alpha-stride; each list sorted by idx ascending

  idx_t num_lists() const;
  idx_t num_complete() const;                       // count of lists with size == alpha
  parlay::slice<const Entry*, const Entry*> list(idx_t k) const;
  idx_t locate(dist_t r) const;                     // F_i(r): first k with radius[k] <= r; binary search
  void  push_back(const Entry* first, uint8_t sz);  // computes radius; asserts sorted-by-idx, sz <= alpha
  void  truncate_tail();                            // drop every list with size < alpha
  void  build_tail();                               // from the last list, repeatedly remove the farthest
                                                    // (tie rule) and push, until the empty list is pushed
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

  void build_sequential();                 // Phase 1  (Alg. 2)
  void build_parallel();                   // Phase 2  (Alg. 6)

  idx_t                    nearest(const point_type& q) const;                 // original index
  parlay::sequence<idx_t>  knn(const point_type& q, idx_t k) const;            // sorted by distance
  parlay::sequence<idx_t>  range(const point_type& q, dist_t delta) const;     // open ball d < delta

  // accessors for tests/bench
  idx_t n() const; idx_t alpha() const;
  const point_type& point(idx_t i) const;            // s_i
  dist_t dist(idx_t i, idx_t j) const;               // metric(point(i), point(j))
  const parlay::sequence<idx_t>& permutation() const;
  const FingerLists& lists(idx_t i) const;
  const parlay::sequence<idx_t>& control() const;    // C[i], parallel build only
};
```

---

## 3. The random walk (`walk.hpp`) — one routine for construction and all queries

State: focus point `cur`, a candidate set `K` (policy object), a target `q`.

```cpp
// Policy concept (one per use):
//   dist_t radius() const;               // radius of the candidate set
//   void   offer(idx_t j, dist_t dj);    // called with the chosen nxt; inserts iff dj < radius()
//   void   stop(idx_t cur);              // called once when the walk terminates
template <class Metric, class Policy>
void random_walk(const MetricSkipList<Metric>& S, const point_type& q, Policy& K, idx_t cur);
```

```
random_walk(q, K, cur):
  loop:
    dc  = d(s_cur, q)
    r   = dc + max(K.radius(), dc)          // ball around cur that must contain any improvement
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
query exact for any `alpha`.

Policies:

| use          | initial K                         | initial cur | `radius()`           | `offer(j, dj)`                                   |
|--------------|-----------------------------------|-------------|----------------------|--------------------------------------------------|
| build `F_i`  | `F_i[0] = {i+1 .. i+alpha}`       | `i+alpha`   | last list's radius   | if `dj < radius`: new list = last list with farthest replaced by `(j,dj)`, kept sorted by idx; `push_back` |
| nearest      | `{0}`                             | `0`         | `d(s_m, q)`          | if `dj < radius`: `m = j`                         |
| knn(k)       | `{0 .. k-1}` (return all if n ≤ k) | `k-1`       | max dist in K        | if `dj < radius`: replace farthest (tie rule)     |
| range(δ)     | ∅ (check `s_0` explicitly)        | `0`         | `δ` (constant)       | report `j`                                        |

Construction's `stop(cur)` builds the tail and records `C[i] = cur` (used only by the parallel build).
Special case: if `i + alpha >= n`, `F_i[0]` is short; it is already exact, so build the tail and skip the walk.

Phase 1 uses binary search for `locate`. Phase 3 replaces it with
`advance`/`align` (Sec. 4) behind the same interface.

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
| K | `advance` pointers | Phase 3 (seq, Alg. 4/5) and Phase 4 (par, Sec. 5.5 / full version) | C, D, G |

A, C, D, E, G are the algorithmic core; F, H, I, J are plumbing. Existing:
`types.hpp`, `metric.hpp`, `Makefile`, `.gitignore`, ParlayLib submodule.

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
  if r - l + 1 <= alpha:                         // base case: nobody in [l,r] has alpha successors
    for i in [l..r]: C[i] = i; provisional(i, r)
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
    F_i = [ {i+1 .. i+alpha} ];  cur = i + alpha
  else:
    cur = C[i]                                    // resume where the walk was blocked
  random_walk(s_i, BuildPolicy(F_i), cur)         // stop() sets C[i] and builds the tail

provisional(i, r):  F_i = tail-only lists over {i+1 .. r}   (fewer than alpha points)
```

Why this is sound (worth keeping in mind while coding):

* `C[i] > i` always (walks start at `i+alpha`), and `C[i] ∈ [l..m]` when `i`
  is in the left half, so the control structure is a forest rooted at
  `C[i] == i` nodes; height O(log n) whp (Lemma 5.1).
* For `i ∈ [l..m]`, the deepest previous level at which `i` was in a *left*
  half had right endpoint exactly `m` (descending through right halves keeps
  the right endpoint). Hence on resume, the first hop out of `F_{C[i]}` lands
  in `(m, r]`, and every later focus point is in the right half, which is
  already complete. **The only left-half list a resumed walk reads is
  `F_{C[i]}`**, which finished in an earlier layer. Within a layer, resumes
  write disjoint `F_i` and read only finished lists — no synchronization
  beyond the layer barrier.
* Points with `i + alpha > r` never get complete lists at this level but can
  still be *focus points* of other walks, so they need the provisional tail
  (this is the gap in the paper's Alg. 6 line 9; at the top level it also
  gives the last `alpha` points the lists queries need).
* After the walk, the stop point becomes `C[i]` for the next level ("useful
  fact", Sec. 5.2), so control points are never recomputed.
* Paper's base case `C[l] <- l` is read as `C[i] <- i` for all `i` in the range.

ParlayLib pieces: `parlay::par_do`, `parlay::parallel_for`,
`parlay::group_by_index`, `parlay::flatten`/`filter`,
`parlay::random_permutation`. Per-point `FingerLists` are independent objects,
so growing them from different workers is safe.

Instrumentation to keep: control-forest depth per level, layer sizes, walk
lengths, lists per point (expect ≈ α·H_n).

---

## 7. Tests (`tests/`, no framework; `CHECK(cond)` macro, nonzero exit on failure)

| test | what |
|------|------|
| `test_reference` | structural invariants (radii non-increasing, sizes, sorted idx, `idx > i`, cached dists, tail shape) + semantic invariant (§5) on n ≤ 500, α ∈ {1,2,4,8}, L2/L1/Linf, D ∈ {1,2,8}, with duplicates and collinear inputs |
| `test_build_seq` | sequential == reference, exactly; n ∈ {1, 2, α, α+1, 50, 500, 2000} |
| `test_query` | nearest / knn / range == brute force on random queries, queries equal to data points, k ≥ n, δ = 0 |
| `test_build_par` | parallel == sequential, exactly; n up to 10⁵; also under `PARLAY_NUM_THREADS=1`, `SEQ=1`, and `DEBUG=1` (ASan/UBSan); records control-forest depth |
| `test_stats` | lists per point vs α·H_n, walk length vs log n (sanity, loose bounds) |

---

## 8. Benchmarks (`bench/`)

* construction: sequential vs parallel, threads 1..14, n = 10⁴..10⁶ (α small for large n)
* queries: batch of 10⁵ NN / kNN queries via `parallel_for`; throughput
* datasets: uniform 2D/3D/8D, Gaussian clusters (varying expansion rate)
* Phase 3/4: binary-search `locate` vs `advance`/`align`

---

## 9. Phases and commits (conventional commits, signed)

| phase | scope | suggested commits |
|-------|-------|-------------------|
| 0 | scaffold (done, uncommitted), README, this plan | `chore: scaffold project with ParlayLib submodule and Makefile`, `docs: rewrite README and add implementation plan` |
| 1 | A B C D E F H, tests reference/seq/query | `feat(core): finger list storage`, `feat(core): random walk and sequential construction`, `feat(query): nearest, knn, range`, `test: reference builder and invariant checks` |
| 2 | G, test_build_par, bench | `feat(par): divide-and-conquer construction with control forest`, `test(par): parallel build matches sequential`, `bench: construction and query benchmarks` |
| 3 | advance/align, sequential (Alg. 4/5) | `feat(core): advance pointers and align` |
| 4 | advance/align, parallel (Sec. 5.5; fetch arXiv:2606.03129 first) | `feat(par): maintain advance pointers in parallel build` |
| 5 | tuning: α sweep, memory compaction, cache layout | `perf: ...` |

---

## 10. Open decisions

* `range()` is an open ball (`d < delta`) to keep the walk's strict `<`; pass
  `nextafter(delta, inf)` for a closed ball.
* Tail lists get no `advance` pointers (paper); when `F*` is a tail list and a
  `nxt` is found, fall back to a local binary search (rare). Decide in Phase 3.
* Memory compaction (slot-history representation) only if needed in Phase 5.
