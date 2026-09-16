// Construction: sequential (optional) vs parallel, and the structure size.
//   bench_build -n 1000000 -alpha 4 -d 2 -data uniform -rounds 3 [-seq 1] [-base 1024] [-adv 1]
// -adv 1 maintains advance pointers (Alg. 5 / Sec. 5.5); -adv 0 uses binary
// search (Alg. 2 / Alg. 6). Vary PARLAY_NUM_THREADS for scaling
// (bench/scaling.sh).
#include <algorithm>
#include <cstdio>

#include "common.hpp"
#include "mskip/metric_skip_list.hpp"

using namespace mskip;

template <int D>
struct Build {
  static void run(const bench::Args& args) {
    const size_t n = args.num("-n", 100000);
    const idx_t alpha = static_cast<idx_t>(args.num("-alpha", 4));
    const int rounds = static_cast<int>(args.num("-rounds", 3));
    const bool seq = args.num("-seq", n <= 200000) != 0;
    const size_t base = static_cast<size_t>(args.num("-base", MetricSkipList<L2<D>>::kDefaultSeqBase));
    const bool adv = args.num("-adv", 1) != 0;
    const std::string data = args.get("-data", "uniform");
    bench::print_header("build", n, alpha, D, data);
    std::printf("  advance pointers: %s\n", adv ? "yes" : "no (binary search)");
    auto pts = bench::dataset<D>(data, n, 1);
    MetricSkipList<L2<D>> S(pts, alpha);
    double best_seq = 1e300, best_par = 1e300;
    if (seq) {
      for (int r = 0; r < rounds; r++) {
        bench::Timer t;
        S.build_sequential(adv);
        best_seq = std::min(best_seq, t.seconds());
      }
      const BuildStats& st = S.stats();
      std::printf("  sequential      %.3f s   (%.2f M walk steps; align moves/step: focus %.2f, pointers %.2f)\n",
                  best_seq, st.steps / 1e6, double(st.focus_moves) / st.steps, double(st.pointer_moves) / st.steps);
    }
    for (int r = 0; r < rounds; r++) {
      bench::Timer t;
      S.build_parallel(base, adv);
      best_par = std::min(best_par, t.seconds());
    }
    const BuildStats& st = S.stats();
    char speedup[32] = "";
    if (seq) std::snprintf(speedup, sizeof speedup, "  speedup %.2fx", best_seq / best_par);
    std::printf("  parallel        %.3f s   (%.2f M walk steps; align moves/step: focus %.2f, pointers %.2f;"
                " %zu merges, forest depth max %zu)%s\n",
                best_par, st.steps / 1e6, double(st.focus_moves) / st.steps, double(st.pointer_moves) / st.steps,
                st.merges, st.max_forest_depth, speedup);
    size_t lists = 0, bytes = 0;
    for (idx_t i = 0; i < S.n(); i++) {
      const FingerLists& F = S.lists(i);
      lists += F.num_lists();
      bytes += F.entries.size() * sizeof(Entry) + F.adv.size() * sizeof(idx_t) + F.radius.size() * sizeof(dist_t) +
               F.size.size();
    }
    std::printf("  structure       %.1f lists/point, %.1f MB\n", double(lists) / n, bytes / 1048576.0);
  }
};

int main(int argc, char** argv) {
  bench::Args args{argc, argv};
  bench::dispatch_dim<Build>(static_cast<int>(args.num("-d", 2)), args);
  return 0;
}
