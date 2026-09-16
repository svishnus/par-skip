// The skip list works over any metric space: here 128-bit strings under the
// Hamming distance. An exact metric declares slack = 0.
//   custom_metric [n]
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>

#include "mskip/metric_skip_list.hpp"

struct Hamming128 {
  using point_type = std::array<uint64_t, 2>;
  static constexpr mskip::dist_t slack = 0;  // integer distances: the triangle inequality holds exactly
  mskip::dist_t operator()(const point_type& a, const point_type& b) const {
    return static_cast<mskip::dist_t>(std::bitset<64>(a[0] ^ b[0]).count() + std::bitset<64>(a[1] ^ b[1]).count());
  }
};

int main(int argc, char** argv) {
  using namespace mskip;
  const size_t n = argc > 1 ? std::atol(argv[1]) : 50000;
  // 20 random "centers", each point a center with ~5% of its bits flipped
  parlay::random rng(7);
  auto centers = parlay::tabulate(20, [&](size_t c) {
    return Hamming128::point_type{rng.ith_rand(2 * c), rng.ith_rand(2 * c + 1)};
  });
  auto pts = parlay::tabulate(n, [&](size_t i) {
    Hamming128::point_type p = centers[i % 20];
    for (int f = 0; f < 6; f++) {
      const uint64_t r = rng.ith_rand(1000 + i * 8 + f);
      p[r & 1] ^= uint64_t{1} << ((r >> 1) & 63);
    }
    return p;
  });

  MetricSkipList<Hamming128> S(pts, /*alpha=*/8);
  S.build();

  const Hamming128::point_type q = centers[3];
  auto near = S.knn_dist(q, 5);
  std::printf("5 nearest points to center 3 (n=%zu):", n);
  for (const Neighbor& x : near) std::printf(" #%u at Hamming distance %g", x.index, double(x.dist));
  std::printf("\nwithin distance 8 of center 3: %zu points\n", S.range(q, 8).size());

  // exactness check on a few random queries
  size_t bad = 0;
  for (size_t t = 0; t < 20; t++) {
    const Hamming128::point_type qq{rng.ith_rand(5000 + 2 * t), rng.ith_rand(5001 + 2 * t)};
    const Neighbor got = S.nearest_dist(qq);
    const dist_t best = parlay::reduce(parlay::delayed_map(pts, [&](const auto& p) { return Hamming128()(qq, p); }),
                                       parlay::minm<dist_t>());
    bad += got.dist != best;
  }
  std::printf("random queries checked against brute force: %zu wrong\n", bad);
  return bad != 0;
}
