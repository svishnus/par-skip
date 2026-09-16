// Query throughput: a batch of nearest-neighbor and k-NN queries answered in
// parallel over a structure built with build_parallel().
//   bench_query -n 1000000 -alpha 4 -d 2 -data uniform -q 100000 -k 10
#include <cstdio>

#include <parlay/parallel.h>

#include "common.hpp"
#include "mskip/metric_skip_list.hpp"

using namespace mskip;

template <int D>
struct Query {
  static void run(const bench::Args& args) {
    const size_t n = args.num("-n", 100000);
    const idx_t alpha = static_cast<idx_t>(args.num("-alpha", 4));
    const size_t nq = args.num("-q", 100000);
    const idx_t k = static_cast<idx_t>(args.num("-k", 10));
    const std::string data = args.get("-data", "uniform");
    bench::print_header("query", n, alpha, D, data);
    auto pts = bench::dataset<D>(data, n, 1);
    auto qs = bench::dataset<D>(data, nq, 2);
    MetricSkipList<L2<D>> S(pts, alpha);
    bench::Timer tb;
    S.build_parallel();
    std::printf("  build           %.3f s\n", tb.seconds());
    parlay::sequence<idx_t> nn(nq);
    for (int round = 0; round < 2; round++) {  // second round is warm
      bench::Timer t;
      parlay::parallel_for(0, nq, [&](size_t i) { nn[i] = S.nearest(qs[i]); });
      const double s = t.seconds();
      std::printf("  nearest         %zu queries in %.3f s: %.2f M/s, %.2f us each\n", nq, s, nq / s / 1e6,
                  s / nq * 1e6);
    }
    parlay::sequence<size_t> sum(nq);
    bench::Timer t;
    parlay::parallel_for(0, nq, [&](size_t i) { sum[i] = S.knn(qs[i], k).size(); });
    const double s = t.seconds();
    std::printf("  knn k=%-3u       %zu queries in %.3f s: %.2f M/s, %.2f us each\n", k, nq, s, nq / s / 1e6,
                s / nq * 1e6);
    size_t total = 0;
    for (size_t x : sum) total += x;
    if (total != nq * std::min<size_t>(k, n)) std::printf("  (unexpected result sizes)\n");
  }
};

int main(int argc, char** argv) {
  bench::Args args{argc, argv};
  bench::dispatch_dim<Query>(static_cast<int>(args.num("-d", 2)), args);
  return 0;
}
