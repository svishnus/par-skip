// Minimal test support: CHECK(cond) reports and counts failures; finish()
// prints a summary and returns the exit status.
#pragma once

#include <cstdio>
#include <string>

#include "mskip/metric_skip_list.hpp"
#include "mskip/reference.hpp"

namespace check {

inline int failures = 0;
inline int checks = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ::check::checks++;                                                           \
    if (!(cond)) {                                                               \
      ::check::failures++;                                                       \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
    }                                                                            \
  } while (0)

// Like CHECK, with a printf-style context on failure.
#define CHECK_CTX(cond, ...)                                                     \
  do {                                                                           \
    ::check::checks++;                                                           \
    if (!(cond)) {                                                               \
      ::check::failures++;                                                       \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed: ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                                         \
      std::fprintf(stderr, "\n");                                                \
    }                                                                            \
  } while (0)

inline int finish(const char* name) {
  std::printf("%s: %d checks, %d failures\n", name, checks, failures);
  return failures ? 1 : 0;
}

// Structural invariants of F plus the cached distances against the metric.
template <class Metric>
std::string validate_lists(const mskip::MetricSkipList<Metric>& S, mskip::idx_t i, const mskip::FingerLists& F) {
  std::string err = F.validate(i);
  if (!err.empty()) return err;
  for (mskip::idx_t k = 0; k < F.num_lists(); k++)
    for (mskip::idx_t e = 0; e < F.size[k]; e++)
      if (F.begin(k)[e].dist != S.dist(i, F.begin(k)[e].idx))
        return "F_" + std::to_string(i) + " list " + std::to_string(k) + ": cached dist";
  return "";
}

template <class Metric>
std::string validate_point(const mskip::MetricSkipList<Metric>& S, mskip::idx_t i) {
  return validate_lists(S, i, S.lists(i));
}

// Advance pointers by definition: F_i[k].advance[j] is the index of
// F_j(F_i[k].radius), for every entry of every complete list.
template <class Metric>
bool check_advance(const mskip::MetricSkipList<Metric>& S, const char* what) {
  if (!S.has_advance()) {
    std::fprintf(stderr, "%s: structure has no advance pointers\n", what);
    return false;
  }
  if (S.unresolved() != 0) {
    std::fprintf(stderr, "%s: %zu pointers still pending after the build\n", what, S.unresolved());
    return false;
  }
  for (mskip::idx_t i = 0; i < S.n(); i++) {
    const mskip::FingerLists& F = S.lists(i);
    if (!F.has_adv || F.adv.size() != F.entries.size()) {
      std::fprintf(stderr, "%s: F_%u adv array\n", what, i);
      return false;
    }
    for (mskip::idx_t k = 0; k < F.num_complete(); k++)
      for (mskip::idx_t e = 0; e < F.alpha; e++) {
        const mskip::idx_t j = F.begin(k)[e].idx;
        const mskip::idx_t want = S.lists(j).locate(F.radius[k]);
        if (F.adv_begin(k)[e] != want) {
          std::fprintf(stderr, "%s: F_%u[%u].advance[%u] = %u, F_%u(%g) is list %u\n", what, i, k, j,
                       F.adv_begin(k)[e], j, static_cast<double>(F.radius[k]), want);
          return false;
        }
      }
  }
  return true;
}

// Every point of S: structure valid and bit-identical to the reference.
template <class Metric>
bool matches_reference(const mskip::MetricSkipList<Metric>& S, const char* what) {
  for (mskip::idx_t i = 0; i < S.n(); i++) {
    std::string err = validate_point(S, i);
    if (!err.empty()) {
      std::fprintf(stderr, "%s: %s\n", what, err.c_str());
      return false;
    }
    if (!mskip::same_lists(S.lists(i), mskip::reference_lists(S, i))) {
      std::fprintf(stderr, "%s: F_%u differs from the reference (n=%u alpha=%u)\n", what, i, S.n(), S.alpha());
      return false;
    }
  }
  return true;
}

}  // namespace check
