// nearest / knn / range against brute force: random queries, queries equal
// to data points, k >= n, delta = 0, and inputs with many ties.
#include <cmath>
#include <cstdio>
#include <limits>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                const parlay::sequence<typename Metric::point_type>& queries, uint64_t seed, bool advance) {
  MetricSkipList<Metric> S(pts, alpha, Metric(), seed);
  if (advance) S.build_parallel(7, true);   // Alg. 4 queries over a parallel build
  else S.build_sequential(false);           // Alg. 3 queries
  const idx_t n = S.n();
  // original index -> permutation position, to compare with the brute force
  parlay::sequence<idx_t> pos(n);
  for (idx_t i = 0; i < n; i++) pos[S.permutation()[i]] = i;
  char what[128];
  std::snprintf(what, sizeof what, "%s alpha=%u n=%u advance=%d", name, alpha, n, int(advance));
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
    for (idx_t k : {idx_t(0), idx_t(1), idx_t(2), idx_t(5), idx_t(n), idx_t(n + 3)}) {
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
  if (advance) std::printf("  %-22s alpha=%u n=%-5u queries=%zu\n", name, alpha, n, queries.size());
}

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                const parlay::sequence<typename Metric::point_type>& queries, uint64_t seed) {
  run<Metric>(name, pts, alpha, queries, seed, false);
  run<Metric>(name, pts, alpha, queries, seed, true);
}

// Computed float L2 distances can violate the triangle inequality by an ulp:
// here d(a,c) > 2 d(a,b) although d(b,c) < d(a,b). Without the metric's
// slack on the search radius, the walk from a would look in a ball that
// excludes c, stop at a's empty tail list, and report a as the nearest
// neighbor of b; the sequential build of (b, a, c) would also miss c as an
// evictor of b's first list. Found by the Phase 2 review.
static void fp_triangle_regression() {
  const Point<2> a{0.0410199612f, 0.216252118f}, b{0.640225053f, 0.749455512f}, c{1.23943007f, 1.28265893f};
  L2<2> d;
  CHECK(d(a, c) > d(a, b) + d(b, c));  // the premise of the test; skip silently otherwise
  if (!(d(a, c) > d(a, b) + d(b, c))) return;
  for (idx_t alpha : {1u, 2u}) {
    MetricSkipList<L2<2>> S(parlay::sequence<Point<2>>{a, c}, alpha, parlay::sequence<idx_t>{0, 1});
    for (bool advance : {false, true}) {
      S.build_sequential(advance);
      CHECK_CTX(S.nearest(b) == 1, "alpha=%u advance=%d: nearest(b) is not c", alpha, int(advance));
      CHECK_CTX(S.knn(b, 1) == parlay::sequence<idx_t>{1}, "alpha=%u advance=%d: knn", alpha, int(advance));
      CHECK_CTX(S.range(b, std::nextafter(d(b, c), 1.0f)) == parlay::sequence<idx_t>{1}, "alpha=%u: range", alpha);
    }
    MetricSkipList<L2<2>> T(parlay::sequence<Point<2>>{b, a, c}, alpha, parlay::sequence<idx_t>{0, 1, 2});
    T.build_sequential(false);
    CHECK_CTX(check::matches_reference(T, "fp triangle (b, a, c)"), "alpha=%u", alpha);
    T.build_parallel(0, true);
    CHECK_CTX(check::matches_reference(T, "fp triangle (b, a, c) parallel"), "alpha=%u", alpha);
  }
}

int main() {
  fp_triangle_regression();
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
