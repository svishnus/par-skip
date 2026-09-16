// Build a metric skip list over random 3D points and compute the k-nearest-
// neighbor graph with one query per point, in parallel.
//   knn_graph [n] [k] [alpha]
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "mskip/data.hpp"
#include "mskip/metric.hpp"
#include "mskip/metric_skip_list.hpp"

int main(int argc, char** argv) {
  using namespace mskip;
  const size_t n = argc > 1 ? std::atol(argv[1]) : 100000;
  const idx_t k = argc > 2 ? static_cast<idx_t>(std::atoi(argv[2])) : 10;
  const idx_t alpha = argc > 3 ? static_cast<idx_t>(std::atoi(argv[3])) : 8;
  auto timer = [] { return std::chrono::steady_clock::now(); };
  auto secs = [](auto t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

  auto pts = data::uniform<3>(n, /*seed=*/42);  // parlay::sequence<Point<3>>

  auto t0 = timer();
  MetricSkipList<L2<3>> S(pts, alpha);  // shuffles the points; keeps a copy in permuted order
  S.build();                            // parallel, with advance pointers
  const double lists = parlay::reduce(parlay::delayed_tabulate(n, [&](size_t i) { return double(S.lists(static_cast<idx_t>(i)).num_lists()); })) / n;
  std::printf("built n=%zu alpha=%u in %.3f s: %.1f lists/point, %.0f MB, %zu workers\n", n, alpha, secs(t0), lists,
              S.memory_bytes() / 1048576.0, parlay::num_workers());

  // k+1 because a point is its own nearest neighbor; drop it.
  t0 = timer();
  auto graph = parlay::tabulate(n, [&](size_t i) {
    auto nn = S.knn_dist(pts[i], k + 1);
    return parlay::sequence<Neighbor>(nn.begin() + 1, nn.end());
  });
  const double tq = secs(t0);
  const double mean_kth = parlay::reduce(parlay::delayed_map(graph, [&](const auto& g) { return double(g.back().dist); })) / n;
  std::printf("%u-NN graph in %.3f s (%.2f us per point); mean distance to the %u-th neighbor: %.4f\n", k, tq,
              tq / n * 1e6, k, mean_kth);
  std::printf("neighbors of point 0:");
  for (const Neighbor& x : graph[0]) std::printf(" %u (%.4f)", x.index, double(x.dist));
  std::printf("\n");
  return 0;
}
