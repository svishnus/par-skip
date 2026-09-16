// Non-random permutations through the explicit-permutation constructor: the
// whp bounds do not apply, but every builder must still agree with the
// reference and every query must stay exact. Covers the paths a random order
// almost never takes: a point with n - alpha evictors (one long list of
// lists), sorted input with no evictors at all (walks that only hop outward),
// and a control forest that is a chain of depth n / alpha.
#include <cmath>
#include <cstdio>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static MetricSkipList<Metric> identity(const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha) {
  return MetricSkipList<Metric>(pts, alpha, parlay::tabulate(pts.size(), [](size_t i) { return static_cast<idx_t>(i); }));
}

// Builds sequentially and in parallel (both modes, pure D&C) and checks all
// of them against the reference and each other.
template <class Metric>
static void agree(const char* what, MetricSkipList<Metric>& S) {
  S.build_sequential(true);
  CHECK_CTX(check::matches_reference(S, what), "%s", what);
  CHECK_CTX(check::check_advance(S, what), "%s", what);
  auto C = S.control();
  auto K = S.control_list();
  MetricSkipList<Metric> seq = S;
  for (bool advance : {false, true}) {
    S.build_parallel(0, advance);
    CHECK_CTX(S.control() == C && S.control_list() == K, "%s: control points (advance=%d)", what, int(advance));
    bool same = true;
    for (idx_t i = 0; same && i < S.n(); i++) same = same_lists(S.lists(i), seq.lists(i), advance);
    CHECK_CTX(same, "%s: parallel lists differ (advance=%d)", what, int(advance));
    if (advance) CHECK_CTX(check::check_advance(S, what), "%s", what);
  }
}

int main() {
  for (idx_t alpha : {1u, 2u, 4u}) {
    {
      // s_0 = 0, s_j = 1/(j+1): every s_j is an evictor of s_0, so F_0 has
      // n - alpha + 1 complete lists; nobody else has any evictor.
      const size_t n = 2000;
      auto pts = parlay::tabulate(n, [](size_t j) { return Point<1>{j == 0 ? 0.0f : 1.0f / (j + 1)}; });
      auto S = identity<L2<1>>(pts, alpha);
      agree("every point an evictor", S);
      CHECK_CTX(S.lists(0).num_complete() == n - alpha, "F_0 has %u complete lists", S.lists(0).num_complete());
      for (int t = 0; t < 50; t++) {
        parlay::random rng(t);
        const Point<1> q{static_cast<float>(data::unit(rng, t))};
        auto want = brute_knn(S, q, 3);
        auto got = S.knn(q, 3);
        bool same = got.size() == want.size();
        for (size_t e = 0; same && e < got.size(); e++) same = S.permutation()[want[e].idx] == got[e];
        CHECK_CTX(same, "knn on the evictor chain, query %d", t);
      }
    }
    {
      // sorted 1-D input: distances from s_i only grow, no evictors anywhere,
      // every walk hops outward to the end.
      const size_t n = 1500;
      auto pts = parlay::tabulate(n, [](size_t j) { return Point<1>{static_cast<float>(j) / 1500}; });
      auto S = identity<L1<1>>(pts, alpha);
      agree("sorted, no evictors", S);
      for (idx_t i = 0; i + alpha < n; i++) CHECK(S.lists(i).num_complete() == 1);
      auto rev = parlay::tabulate(n, [](size_t i) { return static_cast<idx_t>(1499 - i); });
      MetricSkipList<L1<1>> R(pts, alpha, rev);
      agree("sorted descending", R);
    }
    if (alpha >= 2) {
      // s_j = 2^j: the walk of s_i is blocked at s_{i+alpha} at once, so
      // C[i] = i + alpha and the control forest is alpha chains of length
      // n / alpha (the paper's O(log n) height needs a random order). L1,
      // because L2 squares the coordinates, which overflows float from 2^63.
      const size_t n = 120;
      auto pts = parlay::tabulate(n, [](size_t j) { return Point<1>{std::ldexp(1.0f, static_cast<int>(j))}; });
      auto S = identity<L1<1>>(pts, alpha);
      agree("exponential gaps", S);
      size_t chain = 0;
      for (idx_t i = 0; i + 2 * alpha < n; i++) chain += S.control()[i] == i + alpha;
      CHECK_CTX(chain == n - 2 * alpha, "alpha=%u: %zu of %zu control points are i + alpha", alpha, chain, n - 2 * alpha);
      CHECK_CTX(S.stats().max_forest_depth >= n / (2 * alpha) - 2, "alpha=%u: forest depth %zu", alpha,
                S.stats().max_forest_depth);
      std::printf("  exponential gaps alpha=%u n=%zu: control forest depth %zu\n", alpha, n, S.stats().max_forest_depth);
    }
  }
  return check::finish("test_adversarial");
}
