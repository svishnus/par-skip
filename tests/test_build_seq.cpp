// Sequential construction (Alg. 2) must be bit-identical to the reference
// builder, and its control points must be consistent with the structure.
#include <cstdio>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

template <class Metric>
static void run(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                uint64_t seed) {
  MetricSkipList<Metric> S(pts, alpha, Metric(), seed);
  S.build_sequential(false);  // Alg. 2: binary search, no pointers
  char what[128];
  std::snprintf(what, sizeof what, "%s alpha=%u n=%u", name, alpha, S.n());
  CHECK_CTX(check::matches_reference(S, what), "%s", what);
  {
    // Alg. 5: same lists and control points, plus advance pointers by definition
    MetricSkipList<Metric> A(pts, alpha, Metric(), seed);
    A.build_sequential(true);
    CHECK_CTX(check::matches_reference(A, what), "%s (advance)", what);
    CHECK_CTX(check::check_advance(A, what), "%s", what);
    CHECK_CTX(A.control() == S.control() && A.control_list() == S.control_list(), "%s: control points differ between Alg. 2 and Alg. 5", what);
    CHECK_CTX(A.stats().steps == S.stats().steps, "%s: walk lengths differ between Alg. 2 and Alg. 5", what);
  }
  // control points: C[i] == i iff s_i has fewer than alpha successors;
  // otherwise the walk stopped at a tail list of F_{C[i]} at the radius the
  // last complete list of F_i implies.
  for (idx_t i = 0; i < S.n(); i++) {
    const FingerLists& F = S.lists(i);
    const idx_t c = S.control()[i];
    if (i + alpha > S.n() - 1) {
      CHECK_CTX(c == i && F.num_complete() == 0, "%s i=%u", what, i);
    } else {
      CHECK_CTX(c > i && c < S.n() && F.num_complete() > 0, "%s i=%u C=%u", what, i, c);
      const FingerLists& Fc = S.lists(c);
      const idx_t k = S.control_list()[i];
      const dist_t dc = S.dist(c, i);
      const dist_t r = (dc + std::max(F.radius(F.num_complete() - 1), dc)) * S.radius_slack();
      CHECK_CTX(k == Fc.locate(r) && Fc.size(k) < alpha, "%s i=%u C=%u k=%u", what, i, c, k);
    }
  }
  const BuildStats& st = S.stats();
  std::printf("  %-22s alpha=%u n=%-5u walks=%zu steps/walk=%.2f\n", name, alpha, S.n(), st.walks,
              st.walks ? double(st.steps) / st.walks : 0.0);
}

int main() {
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    for (size_t n : {size_t(1), size_t(2), size_t(alpha), size_t(alpha + 1), size_t(50), size_t(500), size_t(2000)}) {
      run<L2<2>>("uniform L2 2D", data::uniform<2>(n, n), alpha, n);
      run<L1<3>>("grid L1 3D (ties)", data::grid<3>(n, 5, n), alpha, n + 1);
    }
    run<Linf<8>>("uniform Linf 8D", data::uniform<8>(1000, 3), alpha, 3);
    run<L2<2>>("clusters L2 2D", data::gaussian_clusters<2>(1500, 8, 0.02, 4), alpha, 4);
    run<L2<2>>("duplicates x3 L2 2D", data::with_duplicates<2>(data::uniform<2>(300, 5), 3), alpha, 5);
    run<L2<1>>("collinear L2 1D", data::collinear<1>(700, 6), alpha, 6);
  }
  return check::finish("test_build_seq");
}
