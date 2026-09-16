// The parallel build must be bit-identical to the sequential one: every
// finger list, every control point and focus-list index. Run also with
// PARLAY_NUM_THREADS=1, SEQ=1 and DEBUG=1 (make test-all).
#include <cstdio>
#include <cstdlib>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                uint64_t seed, bool against_reference = false) {
  for (bool advance : {false, true}) {
  MetricSkipList<Metric> seq(pts, alpha, Metric(), seed);
  seq.build_sequential(advance);
  const BuildStats& ss = seq.stats();
  for (size_t base : {size_t(0), size_t(7), MetricSkipList<Metric>::kDefaultSeqBase}) {
    MetricSkipList<Metric> par(pts, alpha, Metric(), seed);
    par.build_parallel(base, advance);
    char what[160];
    std::snprintf(what, sizeof what, "%s alpha=%u n=%u base=%zu advance=%d", name, alpha, par.n(), base, int(advance));
    CHECK_CTX(par.permutation() == seq.permutation(), "%s: permutations differ", what);
    if (advance) CHECK_CTX(check::check_advance(par, what), "%s", what);
    size_t bad = 0;
    for (idx_t i = 0; i < par.n() && bad < 3; i++) {
      if (!same_lists(par.lists(i), seq.lists(i), advance)) {
        bad++;
        CHECK_CTX(false, "%s: F_%u differs (par %u lists, seq %u lists)", what, i, par.lists(i).num_lists(),
                  seq.lists(i).num_lists());
      }
      if (par.control()[i] != seq.control()[i] || par.control_list()[i] != seq.control_list()[i]) {
        bad++;
        CHECK_CTX(false, "%s: control of %u: par (%u,%u) seq (%u,%u)", what, i, par.control()[i],
                  par.control_list()[i], seq.control()[i], seq.control_list()[i]);
      }
    }
    CHECK(bad == 0);
    if (against_reference) CHECK_CTX(check::matches_reference(par, what), "%s", what);
    const BuildStats& st = par.stats();
    // Every walk of the sequential build is resumed at least once in the
    // parallel one (never fewer iterations overall).
    CHECK_CTX(st.steps >= ss.steps, "%s: steps par=%zu seq=%zu", what, st.steps, ss.steps);
    // the pointer paths of the parallel build were exercised
    if (advance && par.n() >= 1000)
      CHECK_CTX(st.pointers_deferred > 0 && st.pointers_refixed > st.pointers_kept && st.pointers_kept > 0,
                "%s: deferred=%zu refixed=%zu kept=%zu", what, st.pointers_deferred, st.pointers_refixed,
                st.pointers_kept);
    if (base == 0)
      std::printf("  %-20s alpha=%u n=%-6u adv=%d merges=%-6zu forest depth: max=%zu mean=%.2f  steps par/seq=%.2f"
                  "  align moves/step: focus par=%.2f seq=%.2f, pointers par=%.2f seq=%.2f\n",
                  name, alpha, par.n(), int(advance), st.merges, st.max_forest_depth,
                  st.merges ? double(st.layers) / st.merges : 0.0, ss.steps ? double(st.steps) / ss.steps : 0.0,
                  st.steps ? double(st.focus_moves) / st.steps : 0.0, ss.steps ? double(ss.focus_moves) / ss.steps : 0.0,
                  st.steps ? double(st.pointer_moves) / st.steps : 0.0, ss.steps ? double(ss.pointer_moves) / ss.steps : 0.0);
  }
  }
}

// Many seeds, tiny n, every small base case: the merges then happen at every
// possible range shape (ranges of 1..3 points, halves smaller than alpha,
// forests that are all roots). Bit-identical to the sequential Alg. 5 build,
// pointers included.
static void stress() {
  size_t configs = 0;
  for (uint64_t seed = 1; seed <= 6; seed++)
    for (size_t n : {1u, 2u, 3u, 5u, 8u, 13u, 21u, 34u, 47u})
      for (idx_t alpha : {1u, 2u, 3u, 5u, 8u})
        for (size_t base : {1u, 2u, 3u, 5u, 9u}) {
          auto pts = data::uniform<2>(n, seed * 1000 + n);
          MetricSkipList<L2<2>> seq(pts, alpha, L2<2>(), seed);
          seq.build_sequential(true);
          MetricSkipList<L2<2>> par(pts, alpha, L2<2>(), seed);
          par.build_parallel(base, true);
          bool same = par.control() == seq.control() && par.control_list() == seq.control_list() && par.unresolved() == 0;
          for (idx_t i = 0; same && i < par.n(); i++) same = same_lists(par.lists(i), seq.lists(i), true);
          CHECK_CTX(same, "stress: seed=%llu n=%zu alpha=%u base=%zu", (unsigned long long)seed, n, alpha, base);
          if (!same) return;
          configs++;
        }
  std::printf("  stress: %zu configurations bit-identical\n", configs);
}

int main() {
  stress();
  // the limits of the layout: alpha = 255 (uint8_t sizes, full stack buffers) and alpha = 200
  run<L2<2>>("alpha 255", data::uniform<2>(300, 21), 255, 21, true);
  run<L2<3>>("alpha 200", data::uniform<3>(260, 22), 200, 22, true);
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    for (size_t n : {size_t(1), size_t(alpha), size_t(alpha + 1), size_t(2 * alpha + 1), size_t(33), size_t(300)})
      run<L2<2>>("uniform L2 2D", data::uniform<2>(n, n), alpha, n, true);
    run<L1<2>>("grid L1 2D (ties)", data::grid<2>(300, 5, 1), alpha, 1, true);
    run<L2<2>>("duplicates x4", data::with_duplicates<2>(data::uniform<2>(100, 2), 4), alpha, 2, true);
  }
  run<L2<2>>("empty", data::uniform<2>(0, 0), 4, 0, true);
  run<L2<2>>("all identical", parlay::sequence<Point<2>>(300, Point<2>{0.25f, 0.75f}), 4, 11, true);
  run<L1<1>>("1D integers (ties)", data::grid<1>(500, 7, 12), 4, 12, true);
  {
    // rebuilding in place (any mode after any mode) gives the same structure
    auto pts = data::uniform<2>(3000, 13);
    MetricSkipList<L2<2>> S(pts, 4, L2<2>(), 13);
    MetricSkipList<L2<2>> fresh(pts, 4, L2<2>(), 13);
    fresh.build_sequential(true);
    S.build_parallel(0, false);
    S.build_sequential(false);
    S.build_parallel(7, true);
    S.build_sequential(true);
    S.build_parallel(0, true);
    bool same = S.control() == fresh.control() && S.control_list() == fresh.control_list();
    for (idx_t i = 0; same && i < S.n(); i++) same = same_lists(S.lists(i), fresh.lists(i), true);
    CHECK_CTX(same, "rebuild in place differs from a fresh build");
    CHECK(check::check_advance(S, "rebuild"));
  }
  run<L2<2>>("uniform L2 2D", data::uniform<2>(100000, 3), 4, 3);
  run<L2<3>>("uniform L2 3D", data::uniform<3>(30000, 4), 8, 4);
  run<Linf<8>>("uniform Linf 8D", data::uniform<8>(20000, 5), 4, 5);
  run<L2<2>>("clusters L2 2D", data::gaussian_clusters<2>(50000, 20, 0.01, 6), 4, 6);
  run<L1<3>>("grid L1 3D (ties)", data::grid<3>(30000, 12, 7), 4, 7);
  run<L2<1>>("collinear L2 1D", data::collinear<1>(30000, 8), 4, 8);
  run<L2<2>>("duplicates x8", data::with_duplicates<2>(data::uniform<2>(4000, 9), 8), 4, 9);
  return check::finish("test_build_par");
}
