# Parallel Metric Skip Lists and Nearest Neighbor Search

[![ci](https://github.com/svishnus/par-skip/actions/workflows/ci.yml/badge.svg)](https://github.com/svishnus/par-skip/actions/workflows/ci.yml)

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

## Using the library

Header-only C++17 on top of ParlayLib. Include `mskip/metric_skip_list.hpp`
(and `mskip/metric.hpp` for the built-in metrics); everything is in
`namespace mskip`.

```cpp
#include "mskip/metric.hpp"
#include "mskip/metric_skip_list.hpp"

using namespace mskip;
parlay::sequence<Point<2>> pts = ...;            // Point<D> = std::array<float, D>
MetricSkipList<L2<2>> S(pts, /*alpha=*/4);      // shuffles the points (seed 0), keeps a copy
S.build();                                       // parallel; == build_parallel()

idx_t i      = S.nearest(q);                     // index into pts
Neighbor nn  = S.nearest_dist(q);                // {index, dist}
auto ten     = S.knn_dist(q, 10);                // sequence<Neighbor>, by distance then priority
auto ball    = S.range(q, 0.01f);                // indices with d < 0.01, discovery order

parlay::parallel_for(0, qs.size(), [&](size_t t) { out[t] = S.knn(qs[t], 10); });  // queries are const
```

* **Metric**: any type with `point_type`, `dist_t operator()(a, b) const`
  and `static constexpr dist_t slack` — a bound on how far the *computed*
  distances can break the triangle inequality (rounding); the walks enlarge
  their search balls by `1 + slack`, which is all the exactness argument
  needs. `L2`, `L1`, `Linf` accumulate in double and round once, so 4 ulps
  suffice; an exact metric (integers, Hamming, edit distance) uses 0 — see
  [`examples/custom_metric.cpp`](examples/custom_metric.cpp). Distances must
  be finite, non-negative and symmetric.
* **alpha** trades memory for walk length: every query and construction walk
  is exact for any `alpha >= 1`; the paper's log-n bounds need `alpha` to
  grow with the expansion rate (≈ 2^D for uniform data in D dimensions).
  Memory is Θ(α² ln n) per point (≈ 1.5 KB at α = 4, 7 KB at α = 8, times
  ≈ 1.9 resident). 4–8 is a good range in 2D/3D.
* **Builders**: `build_parallel(seq_base, advance)` (Alg. 6 + pointers) and
  `build_sequential(advance)` (Alg. 5 / Alg. 2) produce bit-identical
  structures; `advance = false` drops the advance pointers (36 % less memory,
  ≈ 30 % faster parallel build, ≈ 10 % slower nearest-neighbor queries).
  Rebuilding replaces the structure. `stats()` reports walks, steps, align
  moves, control-forest depth; `memory_bytes()` the logical size.
* **Errors**: `std::invalid_argument` for `alpha` outside `[1, 255]` or too
  many points for `uint32_t`; `std::logic_error` for a query before a build;
  `std::out_of_range` for `nearest` on an empty set.
* **Threads**: queries are const and safe to run concurrently after a build;
  the number of workers is ParlayLib's (`PARLAY_NUM_THREADS`).
* **Determinism**: the permutation is a pure function of `(n, seed)` and the
  structure a pure function of the permutation and the computed distances,
  so builds are reproducible for any number of workers; pass a second
  constructor argument `parlay::sequence<idx_t> perm` to fix the order
  yourself. The bounds assume a random permutation — use a seed the input
  cannot anticipate if the input is untrusted.

### CMake

```cmake
find_package(mskip CONFIG REQUIRED)            # after cmake --install
target_link_libraries(app PRIVATE mskip::mskip)
```
or vendor it:
```cmake
add_subdirectory(par-skip)                      # or FetchContent; needs the submodule
target_link_libraries(app PRIVATE mskip::mskip)
```
ParlayLib is taken from an installed `Parlay` package when one is found
(`find_package(Parlay)`), otherwise from the `external/parlaylib` submodule,
which `cmake --install` then installs into the same prefix.
[`examples/consumer`](examples/consumer) is a complete consumer project.
The Makefile below is the quick path for hacking on the repo itself.

## Build and test

Header-only; ParlayLib is a git submodule. Apple clang / clang++ ≥ 15 or
g++ ≥ 11 with C++17.

```sh
git submodule update --init
make test            # build and run all tests (-O3 -march=native)
make test-all        # also with PARLAY_NUM_THREADS=1, SEQ=1 and DEBUG=1 (ASan/UBSan)
make TSAN=1 test     # ThreadSanitizer (-O1 -g)
make bench           # build benchmarks into build/bench/
make examples        # build examples into build/examples/
make SEQ=1 test      # PARLAY_SEQUENTIAL: single-threaded, easier to debug
make DEBUG=1 test    # -O0 -g with ASan/UBSan
PARLAY_NUM_THREADS=4 build/tests/<name>   # control worker count

cmake -S . -B build/cmake && cmake --build build/cmake -j && ctest --test-dir build/cmake
```

CI (`.github/workflows/ci.yml`) runs the tests with g++ and clang on Linux
and with clang on macOS, both through the Makefile and through CMake, and
installs the package and builds `examples/consumer` against it.

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
| structure (logical / resident) | 44 lists/point, 0.31 / 0.63 GB | 0.44 / 0.85 GB |

| n = 10⁶ | binary search | advance pointers |
|---|---|---|
| sequential build | 33.0 s | 31.6 s |
| parallel build, 1 worker | 19.8 s | 24.0 s |
| parallel build, 2 / 4 / 8 workers | 8.3 / 4.3 / 2.3 s | |
| parallel build, 14 workers | 1.20 s (27× over sequential) | 1.71 s (18×) |
| nearest neighbor, 10⁵ queries in parallel | 1.25 M/s | 1.31 M/s |
| 10-NN, 10⁵ queries in parallel | 0.19 M/s | 0.20 M/s |
| structure (logical / resident) | 50 lists/point, 1.8 / 3.3 GB | 2.5 / 4.8 GB |

The parallel algorithm on one worker beats the sequential one (the
divide-and-conquer order is cache-friendlier), and scaling flattens beyond 8
workers: the build is bound by memory traffic over the structure. The
1-, 2-, 4- and 8-worker rows predate the per-point reservation of the list
arrays (`FingerLists::reserve`), which took the 14-worker build from 2.0 to
1.2 s and the resident set from 3.3× to 1.9× the logical size.

Notes from the measurements (details in `docs/PLAN.md` §9):

* The construction walk visits every evictor, so its length is Θ(α ln n):
  ≈ 77 steps per point at n = 4·10⁵, α = 4.
* Advance pointers make locating the focus list ≈ 2.4 align moves per step
  instead of a binary search over ≈ 40 radii, at the price of ≈ 4 pointer
  moves per step; sequentially that is a wash, in parallel the fix-up passes
  cost ≈ 45 %. Nearest-neighbor queries gain ≈ 10 %.
* The α-stride layout costs ≈ 1.5 KB per point at α = 4 (≈ 7 KB at α = 8),
  and the time per walk step doubles between n = 2.5·10⁴ and 4·10⁵ as the
  structure leaves the caches. The resident set is ≈ 1.9× the logical size
  (ParlayLib's pool allocator hands out power-of-two blocks and keeps freed
  ones; the arrays are reserved per point to avoid the growth chain).
  Compaction of the layout itself is the next tuning step.
* With α ≤ 8 the walks are far longer than log n in 3D and 8D: the analysis
  needs α ≥ 16c³ with c ≈ 2^D. Queries stay exact; only the cost bound is
  lost.
* The same happens for a query far from all the data: with `-data clusters`
  (20 tight Gaussian clusters) and uniform query points, a nearest-neighbor
  query takes ≈ 200 µs instead of 0.4 µs, because the search ball of radius
  2·d(cur, q) contains whole clusters and every step is an outward hop. The
  expansion-rate assumption is about the data *and* the queries.

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
  query.hpp                  nearest, knn, range (+ _dist variants)
  reference.hpp, data.hpp    definition-literal builder and oracles; seeded generators
examples/                    knn_graph, custom_metric (Hamming), consumer (installed package)
tests/                       correctness tests; `make test`, `make test-all`, ctest
bench/                       bench_build, bench_query, scaling.sh
docs/PLAN.md                 design, invariants, and the deviations from the paper
CMakeLists.txt, cmake/       package: mskip::mskip, depends on Parlay::parlay
external/                    ParlayLib submodule
```

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/); commits are signed (`git commit -S`).
- The three builders (reference, sequential, parallel) must produce identical
  structures for the same permutation; see the tie rule in `docs/PLAN.md`.
- MIT license (`LICENSE`); the paper is CC BY 4.0.
