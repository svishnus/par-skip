// The reference builder against the definition: structural invariants and
// the semantic invariant F_i(r) = alpha highest-priority points within r,
// checked exactly (ties included) on small inputs with awkward distances.
#include <cmath>
#include <cstdio>
#include <limits>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                uint64_t seed) {
  MetricSkipList<Metric> S(pts, alpha, Metric(), seed);
  parlay::random rng(seed + 17);
  size_t lists = 0;
  for (idx_t i = 0; i < S.n(); i++) {
    FingerLists F = reference_lists(S, i);
    lists += F.num_lists();
    std::string err = check::validate_lists(S, i, F);
    CHECK_CTX(err.empty(), "%s alpha=%u: %s", name, alpha, err.c_str());
    if (!err.empty()) return;
    // radii of F_i, values just below each radius, and random radii
    parlay::sequence<dist_t> probes;
    for (idx_t k = 0; k < F.num_lists(); k++) {
      probes.push_back(F.radius[k]);
      if (F.radius[k] > 0) probes.push_back(std::nextafter(F.radius[k], dist_t(0)));
    }
    for (int t = 0; t < 8; t++) probes.push_back(static_cast<dist_t>(F.radius[0] * 1.5 * data::unit(rng, i * 8 + t)));
    probes.push_back(std::numeric_limits<dist_t>::infinity());
    for (dist_t r : probes) {
      const idx_t k = F.locate(r);
      CHECK(k < F.num_lists() && F.radius[k] <= r && (k == 0 || F.radius[k - 1] > r));
      const idx_t from = static_cast<idx_t>(rng.ith_rand(i + 7919 * static_cast<size_t>(r * 1e6)) % F.num_lists());
      CHECK_CTX(F.align(from, r) == k, "%s i=%u: align(%u, %g) != locate", name, i, from, static_cast<double>(r));
      auto want = ball_prefix(S, i, r);
      bool same = want.size() == F.size[k];
      for (idx_t e = 0; same && e < F.size[k]; e++) same = F.begin(k)[e].idx == want[e];
      CHECK_CTX(same, "%s alpha=%u i=%u r=%g: F_i(r) != alpha highest-priority points in the ball", name,
                alpha, i, static_cast<double>(r));
      if (!same) return;
    }
  }
  std::printf("  %-28s alpha=%u n=%u  lists/point=%.1f\n", name, alpha, S.n(), double(lists) / S.n());
}

int main() {
  const size_t n = 400;
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    run<L2<2>>("uniform L2 2D", data::uniform<2>(n, 1), alpha, 1);
    run<L1<2>>("uniform L1 2D", data::uniform<2>(n, 2), alpha, 2);
    run<Linf<8>>("uniform Linf 8D", data::uniform<8>(n, 3), alpha, 3);
    run<L2<1>>("uniform L2 1D", data::uniform<1>(n, 4), alpha, 4);
    run<L2<2>>("grid L2 2D (ties)", data::grid<2>(n, 6, 5), alpha, 5);
    run<L1<2>>("grid L1 2D (ties)", data::grid<2>(n, 6, 6), alpha, 6);
    run<Linf<2>>("grid Linf 2D (ties)", data::grid<2>(n, 4, 7), alpha, 7);
    run<L2<3>>("collinear L2 3D", data::collinear<3>(n, 8), alpha, 8);
    run<L2<2>>("duplicates x4 L2 2D", data::with_duplicates<2>(data::uniform<2>(n / 4, 9), 4), alpha, 9);
    run<L2<2>>("all identical", parlay::sequence<Point<2>>(n / 8, Point<2>{0.5f, 0.5f}), alpha, 10);
    run<L2<2>>("clusters L2 2D", data::gaussian_clusters<2>(n, 5, 0.01, 11), alpha, 11);
    for (size_t small : {0u, 1u, 2u, alpha, alpha + 1u})
      run<L2<2>>("tiny", data::uniform<2>(small, 12), alpha, 12);
  }
  return check::finish("test_reference");
}
