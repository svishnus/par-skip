# Parallel Metric Skip Lists and Nearest Neighbor Search

A C++17 / [ParlayLib](https://github.com/cmuparlay/parlaylib) implementation
of the metric skip list and its parallel construction from

> Xiangyun Ding, Rohin Garg, Yan Gu, Yihan Sun.
> *Parallel Metric Skip Lists and Nearest Neighbor Search.* SPAA '26.
> Conference version: [`paper.pdf`](paper.pdf) (CC BY 4.0).
> Full version: [arXiv:2606.03129](https://arxiv.org/abs/2606.03129).

The metric skip list supports exact nearest-neighbor, k-NN, and range queries
in general metric spaces, assuming only a constant expansion rate (no bounded
aspect ratio). Each point keeps a sequence of *finger lists*: for every radius,
the α highest-priority points inside that ball. Queries and construction are
both random walks over these lists.

Implemented here:

| | paper | this repo |
|---|---|---|
| sequential construction, binary search | Alg. 2 | `build_sequential(false)` |
| sequential construction, advance pointers + align | Alg. 5 | `build_sequential(true)` |
| parallel divide-and-conquer with the control forest | Alg. 6 | `build_parallel(seq_base, false)` |
| parallel construction maintaining advance pointers | Sec. 5.5 / App. B | `build_parallel(seq_base, true)` (see [`docs/PLAN.md`](docs/PLAN.md) §6.1 for how it differs from the paper's advance tree) |
| nearest neighbor, k-NN, range | Alg. 3 / Alg. 4 | `nearest`, `knn`, `range` (both navigation modes) |

All three builders (a definition-literal reference, sequential, parallel)
produce bit-identical structures for the same permutation, including the
advance pointers and the control points, and every query is exact for any
α ≥ 1 (the paper's α = 16c³ only enters the running-time analysis). See
[`docs/PLAN.md`](docs/PLAN.md) for the invariants, the tie rule, the module
contracts, the gaps in the paper's pseudocode and how they were filled.

## Usage

```cpp
#include "mskip/data.hpp"              // seeded generators (optional)
#include "mskip/metric_skip_list.hpp"

using namespace mskip;
auto pts = data::uniform<2>(1'000'000);           // parlay::sequence<Point<2>>
MetricSkipList<L2<2>> S(pts, /*alpha=*/4);        // shuffles with seed 0
S.build_parallel();                               // advance pointers on by default
idx_t nn = S.nearest(Point<2>{0.5f, 0.5f});       // index into pts
auto ten = S.knn(pts[7], 10);                     // by distance, ties by priority
auto ball = S.range(pts[7], 0.01f);               // open ball d < 0.01
```

A metric is any type with `point_type`,
`dist_t operator()(const point_type&, const point_type&) const` and a
`static constexpr dist_t slack` bounding how far its *computed* distances can
break the triangle inequality (rounding); the walks enlarge their search
balls by `1 + slack`, which is all the exactness argument needs. `L2`, `L1`,
`Linf` over `std::array<float, D>` are provided in `metric.hpp`; use 0 for
exact metrics. Distances must be finite, non-negative and symmetric.
`MetricSkipList` throws `std::invalid_argument` for `alpha` outside
`[1, 255]`, and a second constructor takes an explicit permutation. The size
and time bounds assume a random permutation, so pass a seed the input cannot
anticipate if the input is untrusted.

## Build and test

Header-only; ParlayLib is a git submodule. Apple clang / clang++ ≥ 15 or
g++ ≥ 11 with C++17.

```sh
git submodule update --init
make test            # build and run all tests (-O3 -march=native)
make test-all        # also with PARLAY_NUM_THREADS=1, SEQ=1 and DEBUG=1 (ASan/UBSan)
make TSAN=1 test     # ThreadSanitizer (-O1 -g)
make bench           # build benchmarks into build/bench/
make SEQ=1 test      # PARLAY_SEQUENTIAL: single-threaded, easier to debug
make DEBUG=1 test    # -O0 -g with ASan/UBSan
PARLAY_NUM_THREADS=4 build/tests/<name>   # control worker count
```

Tests (no framework, `CHECK` macro): `test_reference` checks the structural and
semantic invariants exactly, ties included; `test_build_seq` checks the
sequential build against the reference; `test_build_par` checks the parallel
build against the sequential one for n up to 10⁵ (lists, pointers, control
points); `test_query` checks the queries against brute force in both
navigation modes and holds the rounding regression; `test_adversarial` builds
non-random permutations (every point an evictor, sorted input, a control
forest of depth n/α); `test_stats` checks lists per point, walk length and
control-forest depth against the paper's bounds.

## Measurements

Apple M4 Pro (14 cores), uniform points in [0,1)², L2, α = 4, seed 0.

```sh
build/bench/bench_build -n 200000 -alpha 4 -d 2 -adv 0      # Alg. 2 vs Alg. 6
build/bench/bench_build -n 200000 -alpha 4 -d 2 -adv 1      # Alg. 5 vs Alg. 6 + pointers
build/bench/bench_query -n 200000 -alpha 4 -d 2 -q 100000
bench/scaling.sh -n 1000000 -alpha 4                        # PARLAY_NUM_THREADS = 1 2 4 8 14
```

| n = 2·10⁵ | binary search | advance pointers |
|---|---|---|
| sequential build | 2.28 s | 2.28 s |
| parallel build (14 cores) | 0.18 s (12.8×) | 0.26 s (8.8×) |
| nearest neighbor, 10⁵ queries in parallel | 2.5 M/s | 2.9 M/s |
| 10-NN, 10⁵ queries in parallel | 0.39 M/s | 0.36 M/s |
| structure | 44 lists/point, 309 MB | 443 MB |

| n = 10⁶ | binary search | advance pointers |
|---|---|---|
| sequential build | 33.0 s | 31.6 s |
| parallel build, 1 worker | 19.8 s | 24.0 s |
| parallel build, 2 / 4 / 8 workers | 8.3 / 4.3 / 2.3 s | |
| parallel build, 14 workers | 2.03 s (16.3× over sequential) | 3.12 s (10.1×) |
| nearest neighbor, 10⁵ queries in parallel | 1.25 M/s | 1.31 M/s |
| 10-NN, 10⁵ queries in parallel | 0.19 M/s | 0.20 M/s |
| structure | 50 lists/point, 1.77 GB | 2.54 GB |

The parallel algorithm on one worker beats the sequential one (the
divide-and-conquer order is cache-friendlier), and scaling flattens beyond 8
workers: the build is bound by memory traffic over the 1.8 GB structure.

Notes from the measurements (details in `docs/PLAN.md` §9):

* The construction walk visits every evictor, so its length is Θ(α ln n):
  ≈ 77 steps per point at n = 4·10⁵, α = 4.
* Advance pointers make locating the focus list ≈ 2.4 align moves per step
  instead of a binary search over ≈ 40 radii, at the price of ≈ 4 pointer
  moves per step; sequentially that is a wash, in parallel the fix-up passes
  cost ≈ 45 %. Nearest-neighbor queries gain ≈ 10 %.
* The α-stride layout costs ≈ 1.5 KB per point at α = 4 (≈ 7 KB at α = 8),
  and the time per walk step doubles between n = 2.5·10⁴ and 4·10⁵ as the
  structure leaves the caches. Compaction is the next thing to do (Phase 5).
* With α ≤ 8 the walks are far longer than log n in 3D and 8D: the analysis
  needs α ≥ 16c³ with c ≈ 2^D. Queries stay exact; only the cost bound is
  lost.

### Editor support (clangd)

`compile_commands.json` is generated at the repo root by every `make`
(or `make compile_commands`) with one entry per source *and* per header, so
clangd parses headers with the right flags. `.clangd` only tunes diagnostics.
Regenerate after adding files.

## Layout

```
include/mskip/
  types.hpp, metric.hpp      scalar types; L2/L1/Linf over std::array<float, D>
  finger_list.hpp            FingerLists: alpha-stride storage, locate/align, tail, validate
  metric_skip_list.hpp       MetricSkipList: permutation, lists, accessors (includes the rest)
  walk.hpp                   random_walk + BinarySearchNav / AdvanceNav
  build_seq.hpp              BuildPolicy (lists + advance pointers), build_sequential
  build_par.hpp              build_parallel: D&C, control forest, pointer fix-up
  query.hpp                  nearest, knn, range
  reference.hpp, data.hpp    definition-literal builder and oracles; seeded generators
tests/                       correctness tests; `make test`, `make test-all`
bench/                       bench_build, bench_query, scaling.sh
docs/PLAN.md                 design, invariants, and the deviations from the paper
external/                    ParlayLib submodule
```

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/); commits are signed (`git commit -S`).
- The three builders (reference, sequential, parallel) must produce identical
  structures for the same permutation; see the tie rule in `docs/PLAN.md`.
