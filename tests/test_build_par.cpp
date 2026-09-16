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
  MetricSkipList<Metric> seq(pts, alpha, Metric(), seed);
  seq.build_sequential();
  const BuildStats& ss = seq.stats();
  for (size_t base : {size_t(0), size_t(7), MetricSkipList<Metric>::kDefaultSeqBase}) {
    MetricSkipList<Metric> par(pts, alpha, Metric(), seed);
    par.build_parallel(base);
    char what[160];
    std::snprintf(what, sizeof what, "%s alpha=%u n=%u base=%zu", name, alpha, par.n(), base);
    CHECK_CTX(par.permutation() == seq.permutation(), "%s: permutations differ", what);
    size_t bad = 0;
    for (idx_t i = 0; i < par.n() && bad < 3; i++) {
      if (!same_lists(par.lists(i), seq.lists(i))) {
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
    if (base == 0)
      std::printf("  %-20s alpha=%u n=%-6u merges=%-6zu forest depth: max=%zu mean=%.2f  steps par/seq=%.2f\n",
                  name, alpha, par.n(), st.merges, st.max_forest_depth,
                  st.merges ? double(st.layers) / st.merges : 0.0, ss.steps ? double(st.steps) / ss.steps : 0.0);
  }
}

int main() {
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    for (size_t n : {size_t(1), size_t(alpha), size_t(alpha + 1), size_t(2 * alpha + 1), size_t(33), size_t(300)})
      run<L2<2>>("uniform L2 2D", data::uniform<2>(n, n), alpha, n, true);
    run<L1<2>>("grid L1 2D (ties)", data::grid<2>(300, 5, 1), alpha, 1, true);
    run<L2<2>>("duplicates x4", data::with_duplicates<2>(data::uniform<2>(100, 2), 4), alpha, 2, true);
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
