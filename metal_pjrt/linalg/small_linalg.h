// Small matrices on the GPU: Cholesky, triangular solve and LU of matrices
// of up to kSmallMatrixMax rows/cols, one GPU thread per matrix (per
// right-hand-side line for triangular solves; kernels/small_linalg.metal).
// No XLA, so small_linalg_test runs them against rt::Device alone; the
// handlers in lapack_ffi.cc take this path when the Use* predicate holds.
//
// Each host LAPACK handler synchronizes the stream, which costs a full GPU
// round trip (and splits the command buffer) per call. Programs that solve
// many tiny systems inside loops (a batched 4x4 solve per level of an
// associative scan, say) paid three round trips per solve; these are
// enqueued on the stream like any kernel, with no synchronization.
//
// f32 only. Layouts as the handlers': Cholesky and triangular solve
// row-major, LU column-major (LAPACK's). The output may alias the input.
#ifndef METAL_PJRT_LINALG_SMALL_LINALG_H_
#define METAL_PJRT_LINALG_SMALL_LINALG_H_

#include <cstdint>

#include "absl/status/status.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace linalg {

inline constexpr int64_t kSmallMatrixMax = 32;

// Whether the small kernels take a call: every matrix dimension in
// [1, kSmallMatrixMax] and every element / thread count below 2^32 (the
// kernels' offsets are 32-bit and would alias other batch elements).
bool UseSmallCholesky(int64_t batch, int64_t n);
// A: k x k with k = left_side ? m : n; B: m x n.
bool UseSmallTrsm(int64_t batch, int64_t m, int64_t n, bool left_side);
bool UseSmallGetrf(int64_t batch, int64_t m, int64_t n);

// out[b] = L (lower) or U = L^T (upper) of the row-major n x n a[b], the
// other triangle zeroed; a matrix that is not positive definite becomes all
// NaN. Only the referenced triangle of a is read.
absl::Status RunSmallCholesky(rt::Device* device, rt::Stream* stream,
                              const float* a, float* out, int64_t batch,
                              int64_t n, bool lower);

// x[b] solves op(A) X = B (left_side) or X op(A) = B, row-major; A is lower
// or upper triangular (the other triangle is not read), op transposes it
// when `transpose`, and its diagonal is taken as 1 when `unit_diagonal`.
absl::Status RunSmallTrsm(rt::Device* device, rt::Stream* stream,
                          const float* a, const float* b, float* x,
                          int64_t batch, int64_t m, int64_t n, bool left_side,
                          bool lower, bool transpose, bool unit_diagonal);

// LU with partial pivoting (the first row of largest magnitude, as sgetrf)
// of the column-major m x n a[b]: lu[b] holds L (unit, below the diagonal)
// and U; pivots[b] (min(m, n), 0-based) the row swaps and permutation[b]
// (m) the row order they produce. Exactly singular matrices are not an
// error (zero pivots stay).
absl::Status RunSmallGetrf(rt::Device* device, rt::Stream* stream,
                           const float* a, float* lu, int32_t* pivots,
                           int32_t* permutation, int64_t batch, int64_t m,
                           int64_t n);

}  // namespace linalg
}  // namespace metal_pjrt

#endif  // METAL_PJRT_LINALG_SMALL_LINALG_H_
