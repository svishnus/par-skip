// nearest / knn / range against brute force: random queries, queries equal
// to data points, k >= n, delta = 0, and inputs with many ties.
#include <cstdio>
#include <limits>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                const parlay::sequence<typename Metric::point_type>& queries, uint64_t seed) {
  MetricSkipList<Metric> S(pts, alpha, Metric(), seed);
  S.build_sequential();
  const idx_t n = S.n();
  // original index -> permutation position, to compare with the brute force
  parlay::sequence<idx_t> pos(n);
  for (idx_t i = 0; i < n; i++) pos[S.permutation()[i]] = i;
  char what[128];
  std::snprintf(what, sizeof what, "%s alpha=%u n=%u", name, alpha, n);
  size_t bad = 0;
  for (size_t t = 0; t < queries.size() && bad < 3; t++) {
    const auto& q = queries[t];
    // nearest: same point as brute force (closest, then highest priority)
    auto nn = brute_knn(S, q, 1);
    const idx_t got = pos[S.nearest(q)];
    if (got != nn[0].idx) {
      bad++;
      CHECK_CTX(false, "%s query %zu: nearest %u (d=%g) vs brute %u (d=%g)", what, t, got,
                double(S.dist_to(got, q)), nn[0].idx, double(nn[0].dist));
    }
    for (idx_t k : {idx_t(1), idx_t(2), idx_t(5), idx_t(n), idx_t(n + 3)}) {
      auto want = brute_knn(S, q, k);
      auto res = S.knn(q, k);
      bool same = res.size() == want.size();
      for (size_t e = 0; same && e < res.size(); e++) same = pos[res[e]] == want[e].idx;
      if (!same) {
        bad++;
        CHECK_CTX(false, "%s query %zu: knn k=%u differs", what, t, k);
      }
    }
    auto d5 = brute_knn(S, q, 5).back().dist;
    for (dist_t delta : {dist_t(0), std::nextafter(nn[0].dist, dist_t(0)), std::nextafter(nn[0].dist, dist_t(1e9)),
                         d5, dist_t(0.05), dist_t(0.3), std::numeric_limits<dist_t>::infinity()}) {
      auto want = brute_range(S, q, delta);
      auto res = S.range(q, delta);
      for (idx_t& x : res) x = pos[x];
      std::sort(res.begin(), res.end());
      if (res != want) {
        bad++;
        CHECK_CTX(false, "%s query %zu: range delta=%g gives %zu points, brute %zu", what, t, double(delta),
                  res.size(), want.size());
      }
    }
  }
  CHECK(bad == 0);
  std::printf("  %-22s alpha=%u n=%-5u queries=%zu\n", name, alpha, n, queries.size());
}

int main() {
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    for (size_t n : {size_t(1), size_t(2), size_t(alpha + 1), size_t(37), size_t(1000)}) {
      auto pts = data::uniform<2>(n, n);
      auto qs = data::uniform<2>(40, n + 100);
      for (size_t i = 0; i < std::min<size_t>(n, 20); i++) qs.push_back(pts[i]);  // queries on data points
      run<L2<2>>("uniform L2 2D", pts, alpha, qs, n);
    }
    {
      auto pts = data::grid<2>(800, 6, 1);
      auto qs = data::grid<2>(30, 6, 2);
      for (auto& q : data::uniform<2>(30, 3)) qs.push_back(q);
      run<L1<2>>("grid L1 2D (ties)", pts, alpha, qs, 1);
      run<Linf<2>>("grid Linf 2D (ties)", pts, alpha, qs, 2);
    }
    run<L2<8>>("clusters L2 8D", data::gaussian_clusters<8>(1200, 6, 0.05, 5), alpha, data::uniform<8>(40, 6), 5);
    run<L2<3>>("duplicates x4 L2 3D", data::with_duplicates<3>(data::uniform<3>(200, 7), 4), alpha,
               data::uniform<3>(40, 8), 7);
    run<L2<1>>("collinear L2 1D", data::collinear<1>(500, 9), alpha, data::uniform<1>(40, 10), 9);
  }
  return check::finish("test_query");
}
