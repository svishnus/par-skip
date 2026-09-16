# Parallel Metric Skip Lists and Nearest Neighbor Search

A C++/[ParlayLib](https://github.com/cmuparlay/parlaylib) implementation of the
metric skip list and its parallel construction from

> Xiangyun Ding, Rohin Garg, Yan Gu, Yihan Sun.
> *Parallel Metric Skip Lists and Nearest Neighbor Search.* SPAA '26.
> Conference version: [`paper.pdf`](paper.pdf) (CC BY 4.0).
> Full version: [arXiv:2606.03129](https://arxiv.org/abs/2606.03129).

The metric skip list supports exact nearest-neighbor, k-NN, and range queries
in general metric spaces, assuming only a constant expansion rate (no bounded
aspect ratio). Each point keeps a sequence of *finger lists*: for every radius,
the α highest-priority points inside that ball. Queries and construction are
both random walks over these lists.

## Roadmap

See [`docs/PLAN.md`](docs/PLAN.md) for the data model, invariants, module
contracts, and test plan.

1. Sequential construction (Alg. 2) and queries, verified against a
   definition-literal reference builder
2. Parallel divide-and-conquer construction with the control forest (Alg. 6),
   verified bit-identical to sequential
3. `advance`/`align` pointers, sequential (Sec. 4)
4. `advance`/`align` pointers, parallel (Sec. 5.5)
5. Tuning and applications from the paper

## Build

Header-only; ParlayLib is a git submodule. Apple clang / clang++ ≥ 15 or
g++ ≥ 11 with C++17.

```sh
git submodule update --init
make test            # build and run all tests (-O3 -march=native)
make bench           # build benchmarks into build/bench/
make SEQ=1 test      # PARLAY_SEQUENTIAL: single-threaded, easier to debug
make DEBUG=1 test    # -O0 -g with ASan/UBSan
PARLAY_NUM_THREADS=4 build/tests/<name>   # control worker count
```

### Editor support (clangd)

`compile_commands.json` is generated at the repo root by every `make`
(or `make compile_commands`) with one entry per source *and* per header, so
clangd parses headers with the right flags. `.clangd` only tunes diagnostics.
Regenerate after adding files.

## Layout

```
include/mskip/   the library (header-only)
tests/           correctness tests, no framework; `make test`
bench/           benchmarks
docs/PLAN.md     design and implementation plan
external/        ParlayLib submodule
```

## Conventions

- [Conventional Commits](https://www.conventionalcommits.org/); commits are signed (`git commit -S`).
- The three builders (reference, sequential, parallel) must produce identical
  structures for the same permutation; see the tie rule in `docs/PLAN.md`.
