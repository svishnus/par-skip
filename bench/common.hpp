// Shared benchmark plumbing: a wall-clock timer, a tiny argument parser and
// the datasets from data.hpp selected by name.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <parlay/parallel.h>

#include "mskip/data.hpp"
#include "mskip/metric.hpp"

namespace bench {

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
};

struct Args {
  int argc;
  char** argv;
  const char* get(const char* key, const char* def) const {
    for (int i = 1; i + 1 < argc; i++)
      if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
  }
  long num(const char* key, long def) const {
    const char* v = get(key, nullptr);
    return v ? std::atol(v) : def;
  }
  double real(const char* key, double def) const {
    const char* v = get(key, nullptr);
    return v ? std::atof(v) : def;
  }
};

// uniform | clusters (20 Gaussian clusters, sigma = 0.01) | grid (side 64)
template <int D>
parlay::sequence<mskip::Point<D>> dataset(const std::string& name, size_t n, uint64_t seed) {
  if (name == "uniform") return mskip::data::uniform<D>(n, seed);
  if (name == "clusters") return mskip::data::gaussian_clusters<D>(n, 20, 0.01, seed);
  if (name == "grid") return mskip::data::grid<D>(n, 64, seed);
  std::fprintf(stderr, "unknown dataset %s (uniform | clusters | grid)\n", name.c_str());
  std::exit(2);
}

// Runs f<D>() for the requested dimension.
template <template <int> class F, class... A>
void dispatch_dim(int d, A&&... a) {
  switch (d) {
    case 2: F<2>::run(std::forward<A>(a)...); break;
    case 3: F<3>::run(std::forward<A>(a)...); break;
    case 8: F<8>::run(std::forward<A>(a)...); break;
    default: std::fprintf(stderr, "dimension must be 2, 3 or 8\n"); std::exit(2);
  }
}

inline void print_header(const char* what, size_t n, unsigned alpha, int d, const std::string& data) {
  std::printf("%s: n=%zu alpha=%u D=%d data=%s workers=%zu\n", what, n, alpha, d, data.c_str(),
              parlay::num_workers());
}

}  // namespace bench
