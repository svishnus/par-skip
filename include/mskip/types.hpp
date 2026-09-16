// Shared scalar types for the metric skip list.
#pragma once

#include <cstdint>

namespace mskip {

// Position of a point in the random permutation (0-based). Lower index means
// higher priority; the paper's s_i is pts[i - 1]. Also used for list indices.
using idx_t = uint32_t;

// Distances. All three builders (reference, sequential, parallel) evaluate the
// same metric on the same points, so the structure is a deterministic function
// of the permutation regardless of floating-point rounding.
using dist_t = float;

}  // namespace mskip
