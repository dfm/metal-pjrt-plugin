// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Batched f32 dense linear algebra on the host with Apple Accelerate (its
// current LAPACK, the `$NEWLAPACK` symbols, LP64, and CBLAS): the work
// behind the metal$cholesky, metal$triangular_solve and metal$lapack_*
// handlers (lapack_ffi.cc), which synchronize the stream and copy the input
// to the output before calling these on the shared-memory buffers. No XLA and no Metal, so
// lapack_host_test is a plain host test.
//
// Every function works in place on `x`, a batch of matrices stored back to
// back, and returns an Internal error only if LAPACK reports an illegal
// argument (a plugin bug). Empty matrices do nothing.
#ifndef METAL_PJRT_LINALG_LAPACK_HOST_H_
#define METAL_PJRT_LINALG_LAPACK_HOST_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace metal_pjrt {
namespace linalg {

// Row-major n x n: L (lower) or U = L^T with the other triangle zeroed; a
// matrix that is not positive definite becomes all NaN (as on JAX's CPU
// backend). Only the referenced triangle is read.
absl::Status HostCholesky(float* x, int64_t batch, int n, bool lower);

// Row-major: x[b] (m x n, holding B) := X with op(A) X = B (left_side) or
// X op(A) = B; a[b] is k x k, k = left_side ? m : n.
void HostTriangularSolve(const float* a, float* x, int64_t batch, int m,
                         int n, bool left_side, bool lower, bool transpose,
                         bool unit_diagonal);

// Column-major m x n (LAPACK's layout, from here on): sgetrf. pivots[b]
// (min(m, n)) are 0-based and permutation[b] (m) is the row order they
// produce. Singular matrices are not an error.
absl::Status HostGetrf(float* x, int32_t* pivots, int32_t* permutation,
                       int64_t batch, int m, int n);

// sgtsv (Gaussian elimination with partial pivoting): x[b] (n x nrhs,
// holding B) := X with A X = B, A[b] tridiagonal with sub-, main and
// super-diagonals dl[b], d[b], du[b] (n each, as JAX passes them: dl[b][0]
// and du[b][n - 1] are ignored). The diagonals are not modified. A singular
// A makes that X all NaN (as on JAX's CPU backend).
absl::Status HostGtsv(const float* dl, const float* d, const float* du,
                      float* x, int64_t batch, int n, int nrhs);

// sgeqrf: R and the Householder vectors in x, taus[b] (min(m, n)).
absl::Status HostGeqrf(float* x, float* taus, int64_t batch, int m, int n);

// sorgqr: the first n columns of Q from geqrf's x and taus[b] (k).
// Requires m >= n >= k.
absl::Status HostOrgqr(float* x, const float* taus, int64_t batch, int m,
                       int n, int k);

// ssyevd with vectors: x[b] (n x n, the `lower` or upper triangle read)
// := the eigenvectors, w[b] (n) the eigenvalues ascending. A failed
// decomposition, or a non-finite value in the triangle read, makes both NaN.
absl::Status HostSyevd(float* x, float* w, int64_t batch, int n, bool lower);

// sgesdd: s[b] (min(m, n)) the singular values, descending; with u and vt
// (non-null) also U (m x m if full_matrices, else m x min(m, n)) and V^T
// (n x n, else min(m, n) x n). x is overwritten. A failed decomposition,
// or a non-finite input, makes s (and u, vt) NaN.
absl::Status HostGesdd(float* x, float* s, float* u, float* vt,
                       int64_t batch, int m, int n, bool full_matrices);

// The workspace size for a float `query` result (LAPACK's lwork = -1 call),
// at least `minimum` (the documented minimum) and 1. Above 2^24 the float
// can round below the size LAPACK then requires, so it is rounded up to the
// next float (as jaxlib does). InvalidArgument above INT_MAX; `name`
// prefixes the message.
absl::StatusOr<int> LapackWorkspace(const char* name, float query,
                                    int64_t minimum);

}  // namespace linalg
}  // namespace metal_pjrt

#endif  // METAL_PJRT_LINALG_LAPACK_HOST_H_
