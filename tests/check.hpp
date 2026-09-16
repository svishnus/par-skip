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
