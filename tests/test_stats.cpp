// Sanity bounds on the structure and the walks (loose, not proofs):
// lists per point ~ alpha * H_n, walk length ~ log n, control-forest depth
// ~ log n.
#include <cmath>
#include <cstdio>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

// The walk-length bound is only checked where alpha is large enough for the
// dimension: the paper's analysis needs alpha >= 16c^3 with c ~ 2^D, and with
// a small alpha most hops are "outward" hops, so walks get long in 3D/8D.
template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                bool bound_walk) {
  MetricSkipList<Metric> S(pts, alpha, Metric(), 1);
  S.build_parallel(0);
  const idx_t n = S.n();
  double lists = 0, complete = 0;
  size_t max_lists = 0;
  for (idx_t i = 0; i < n; i++) {
    lists += S.lists(i).num_lists();
    complete += S.lists(i).num_complete();
    max_lists = std::max<size_t>(max_lists, S.lists(i).num_lists());
  }
  lists /= n;
  complete /= n;
  const double hn = std::log(n) + 0.5772;
  const BuildStats& st = S.stats();
  const double steps = double(st.steps) / st.walks;
  const double log2n = std::log2(n);
  std::printf("  %-18s alpha=%u n=%-6u lists/point=%.1f (alpha*H_n=%.1f, max=%zu)  steps/walk=%.1f (log2 n=%.1f)"
              "  forest depth max=%zu\n",
              name, alpha, n, lists, alpha * hn, max_lists, steps, log2n, st.max_forest_depth);
  // complete lists: the first plus one per evictor; E[#evictors] <= alpha * H_n
  CHECK_CTX(complete <= 1.3 * alpha * hn && complete >= 0.3 * alpha * hn, "%s: %.1f complete lists/point", name, complete);
  CHECK_CTX(max_lists <= 6 * alpha * hn, "%s: max lists %zu", name, max_lists);
  if (bound_walk) CHECK_CTX(steps <= 2 * log2n, "%s: %.1f steps/walk", name, steps);
  CHECK_CTX(st.max_forest_depth <= 6 * log2n, "%s: forest depth %zu", name, st.max_forest_depth);
}

int main() {
  for (idx_t alpha : {2u, 4u, 8u}) {
    run<L2<2>>("uniform L2 2D", data::uniform<2>(50000, 1), alpha, alpha >= 4);
    run<L2<3>>("uniform L2 3D", data::uniform<3>(50000, 2), alpha, alpha >= 8);
    run<L2<8>>("uniform L2 8D", data::uniform<8>(20000, 3), alpha, false);
    run<L2<2>>("clusters L2 2D", data::gaussian_clusters<2>(50000, 10, 0.02, 4), alpha, alpha >= 4);
    run<L1<3>>("grid L1 3D", data::grid<3>(30000, 10, 5), alpha, alpha >= 8);
  }
  return check::finish("test_stats");
}
