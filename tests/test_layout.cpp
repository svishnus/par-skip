// The compact list storage (finger_list.hpp): the structure must not depend
// on the checkpoint stride, every builder must agree bit for bit at every
// stride (pointers included, whose slots depend on the stride), a rebuild
// may change the stride, and the memory must be Θ(alpha ln n) per point.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "check.hpp"
#include "mskip/data.hpp"
#include "mskip/metric.hpp"

using namespace mskip;

// Materialized lists and radii of every point, independent of the layout.
template <class Metric>
static bool same_structure(const MetricSkipList<Metric>& a, const MetricSkipList<Metric>& b) {
  if (a.control() != b.control() || a.control_list() != b.control_list()) return false;
  for (idx_t i = 0; i < a.n(); i++)
    if (!same_lists(a.lists(i), b.lists(i), false)) return false;
  return true;
}

template <class Metric>
static void strides(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha,
                    uint64_t seed, idx_t extra_stride = 0) {
  MetricSkipList<Metric> ref(pts, alpha, Metric(), seed);  // default stride
  ref.build_sequential(true);
  const idx_t max_stride = FingerLists::max_stride(alpha);
  for (idx_t stride : {idx_t(1), idx_t(2), alpha, idx_t(2 * alpha), max_stride, extra_stride}) {
    if (stride == 0 || stride > max_stride) continue;
    char what[128];
    std::snprintf(what, sizeof what, "%s alpha=%u n=%u stride=%u", name, alpha, ref.n(), stride);
    MetricSkipList<Metric> seq(pts, alpha, Metric(), seed);
    seq.set_checkpoint_stride(stride);
    seq.build_sequential(true);
    CHECK_CTX(seq.checkpoint_stride() == stride, "%s", what);
    CHECK_CTX(check::matches_reference(seq, what), "%s", what);
    CHECK_CTX(check::check_advance(seq, what), "%s", what);
    CHECK_CTX(same_structure(seq, ref), "%s: structure depends on the stride", what);
    for (bool advance : {false, true}) {
      MetricSkipList<Metric> par(pts, alpha, Metric(), seed);
      par.set_checkpoint_stride(stride);
      par.build_parallel(7, advance);
      bool same = par.control() == seq.control() && par.control_list() == seq.control_list();
      for (idx_t i = 0; same && i < par.n(); i++) same = same_lists(par.lists(i), seq.lists(i), advance);
      CHECK_CTX(same, "%s advance=%d: parallel differs from sequential", what, int(advance));
      if (advance) CHECK_CTX(check::check_advance(par, what), "%s", what);
    }
  }
  // the stride can change between builds of the same object
  MetricSkipList<Metric> S(pts, alpha, Metric(), seed);
  S.set_checkpoint_stride(1);
  S.build_parallel(0, true);
  S.set_checkpoint_stride(max_stride);
  S.build_parallel(0, true);
  CHECK_CTX(S.checkpoint_stride() == max_stride && same_structure(S, ref) && check::check_advance(S, name),
            "%s: rebuild with another stride", name);
  S.set_checkpoint_stride(0);
  S.build_sequential(false);
  CHECK_CTX(S.checkpoint_stride() == ref.checkpoint_stride() && same_structure(S, ref), "%s: default stride", name);
}

static void limits() {
  MetricSkipList<L2<2>> S(data::uniform<2>(100, 1), 4);
  bool threw = false;
  try {
    S.set_checkpoint_stride(FingerLists::max_stride(4) + 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(FingerLists::max_stride(255) == 2 && FingerLists::default_stride(255) == 2);
  CHECK(FingerLists::default_stride(4) >= 4 && FingerLists::default_stride(200) == 57);
  FingerLists empty;  // default-constructed: no lists, usable
  CHECK(empty.num_lists() == 0 && empty.bytes() == 0 && empty.logical_bytes() == 0);
  empty.build_tail();
  CHECK(empty.num_lists() == 1 && empty.size(0) == 0 && empty.validate(0).empty());
}

// Copies and moves of a built structure answer queries like the original;
// build_tail is idempotent and reserve keeps a built list intact.
static void copies() {
  auto pts = data::uniform<2>(2000, 5);
  auto qs = data::uniform<2>(50, 6);
  MetricSkipList<L2<2>> S(pts, 4, L2<2>(), 5);
  S.build_parallel(7, true);
  MetricSkipList<L2<2>> C = S;  // every list now owns its buffer
  MetricSkipList<L2<2>> A(pts, 2, L2<2>(), 9);
  A.build_sequential(false);
  A = S;  // copy-assignment over a built structure
  MetricSkipList<L2<2>>& alias = A;
  A = alias;  // self-assignment must not read the slab it drops
  MetricSkipList<L2<2>> M = std::move(C);
  bool same = true;
  for (idx_t i = 0; same && i < S.n(); i++)
    same = same_lists(S.lists(i), M.lists(i), true) && same_lists(S.lists(i), A.lists(i), true) && M.lists(i).owns_buffer();
  CHECK_CTX(same, "copies differ from the original");
  for (const auto& q : qs) CHECK(S.knn(q, 5) == M.knn(q, 5) && S.knn(q, 5) == A.knn(q, 5));
  CHECK(check::check_advance(A, "copy") && check::check_advance(M, "move"));
  A.build_parallel(0, true);  // a rebuild of a copy reserves a slab of its own
  CHECK(check::matches_reference(A, "rebuilt copy"));
  FingerLists F = S.lists(0);
  const idx_t lists = F.num_lists();
  F.build_tail();
  CHECK(F.num_lists() == lists);
  F.reserve(4 * F.capacity());
  CHECK(F.num_lists() == lists && same_lists(F, S.lists(0), true) && F.validate(0).empty());
}

// Θ(alpha ln n) per point: at most ~30 bytes per list plus the tail base,
// against the alpha-stride layout's 5 + 12 alpha per list.
template <class Metric>
static void memory(const char* name, const parlay::sequence<typename Metric::point_type>& pts, idx_t alpha) {
  MetricSkipList<Metric> S(pts, alpha);
  S.build_parallel();
  double lists = 0;
  size_t logical = 0, allocated = 0;
  for (idx_t i = 0; i < S.n(); i++) {
    const FingerLists& F = S.lists(i);
    lists += F.num_lists();
    logical += F.logical_bytes();
    allocated += F.bytes();
    CHECK_CTX(F.logical_bytes() <= F.bytes(), "%s: F_%u logical %zu > allocated %zu", name, i, F.logical_bytes(), F.bytes());
  }
  // allocated_bytes counts the whole slab, abandoned slices included
  CHECK(logical == S.memory_bytes() && allocated <= S.allocated_bytes() && S.allocated_bytes() < 1.1 * allocated);
  const double per_point = double(logical) / S.n();
  const double bound = 30.0 * lists / S.n() + 20.0 * alpha;
  CHECK_CTX(per_point <= bound, "%s alpha=%u: %.0f bytes per point, bound %.0f", name, alpha, per_point, bound);
  // the reservation (mean + 2 sigma) leaves a few percent of the points to grow their buffer
  size_t grown = 0;
  for (idx_t i = 0; i < S.n(); i++) grown += S.lists(i).owns_buffer();
  CHECK_CTX(grown < S.n() / 10 && double(allocated) < 1.6 * logical, "%s alpha=%u: %zu of %u points grew, allocated/logical %.2f",
            name, alpha, grown, S.n(), double(allocated) / logical);
  std::printf("  %-16s alpha=%u n=%-6u %.1f lists/point, %.0f B/point logical, %.0f allocated, %.1f%% grown\n", name,
              alpha, S.n(), lists / S.n(), per_point, double(allocated) / S.n(), 100.0 * grown / S.n());
}

int main() {
  limits();
  for (idx_t alpha : {1u, 2u, 4u, 8u}) {
    strides<L2<2>>("uniform L2 2D", data::uniform<2>(3000, alpha), alpha, alpha);
    strides<L1<3>>("grid L1 3D (ties)", data::grid<3>(2000, 5, alpha + 10), alpha, alpha + 10);
    strides<L2<2>>("tiny", data::uniform<2>(alpha + 2, 3), alpha, 3);
  }
  strides<L2<2>>("alpha 255", data::uniform<2>(400, 21), 255, 21);
  strides<L2<2>>("alpha 60", data::uniform<2>(500, 22), 60, 22);
  // the 64-slot boundary of the live mask: block slots 64 (alpha 63/stride 2, 64/1, 33/32) and 65 (65/1)
  strides<L2<2>>("alpha 63", data::uniform<2>(300, 23), 63, 23);
  strides<L2<2>>("alpha 64", data::uniform<2>(300, 24), 64, 24);
  strides<L2<2>>("alpha 65", data::uniform<2>(300, 25), 65, 25);
  strides<L2<2>>("alpha 33", data::uniform<2>(300, 26), 33, 26, 32);
  copies();
  for (idx_t alpha : {2u, 4u, 8u, 16u}) {
    memory<L2<2>>("uniform L2 2D", data::uniform<2>(100000, 1), alpha);
    memory<L2<3>>("clusters L2 3D", data::gaussian_clusters<3>(50000, 10, 0.02, 2), alpha);
  }
  return check::finish("test_layout");
}
