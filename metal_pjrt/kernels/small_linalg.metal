// Small (n <= 32) dense linear algebra on the GPU, one thread per matrix (per
// right-hand-side line for triangular solves); see linalg/small_linalg.h.
// SmallParams must match the C++ struct of the same name in small_linalg.cc.
#include <metal_stdlib>
using namespace metal;
struct SmallParams { uint batch; uint m; uint n; uint k; uint flags; };
constant uint kLower = 1, kLeft = 2, kTrans = 4, kUnit = 8;

// Cholesky of a row-major n x n matrix, in place semantics (a may equal out):
// only the referenced triangle of a is read; the result holds L (lower) or
// U = L^T (upper) with the other triangle zeroed; not positive definite ->
// all NaN. One thread per matrix.
kernel void small_cholesky(device const float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant SmallParams& p [[buffer(2)]],
                           uint b [[thread_position_in_grid]]) {
  if (b >= p.batch) return;
  const uint n = p.n;
  const bool lower = p.flags & kLower;
  device const float* A = a + b * n * n;
  device float* L = out + b * n * n;
  // (i, j) with i >= j: the referenced element of A and the slot of L_ij.
  #define LD(i, j) (lower ? A[(i) * n + (j)] : A[(j) * n + (i)])
  #define ST(i, j) (lower ? L[(i) * n + (j)] : L[(j) * n + (i)])
  bool bad = false;
  for (uint j = 0; j < n && !bad; ++j) {
    float s = LD(j, j);
    for (uint t = 0; t < j; ++t) s -= ST(j, t) * ST(j, t);
    if (!(s > 0.0f)) { bad = true; break; }
    const float d = sqrt(s);
    ST(j, j) = d;
    for (uint i = j + 1; i < n; ++i) {
      float v = LD(i, j);
      for (uint t = 0; t < j; ++t) v -= ST(i, t) * ST(j, t);
      ST(i, j) = v / d;
    }
  }
  #undef LD
  #undef ST
  if (bad) {
    for (uint i = 0; i < n * n; ++i) L[i] = NAN;
    return;
  }
  for (uint i = 0; i < n; ++i) {
    for (uint j = i + 1; j < n; ++j) {
      if (lower) L[i * n + j] = 0.0f; else L[j * n + i] = 0.0f;
    }
  }
}

// Triangular solve, row-major: op(A) X = B (left) or X op(A) = B (right),
// A k x k, B and X m x n (x may equal b). One thread per (matrix, column of X)
// on the left, per (matrix, row of X) on the right.
kernel void small_trsm(device const float* a [[buffer(0)]],
                       device const float* bm [[buffer(1)]],
                       device float* x [[buffer(2)]],
                       constant SmallParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  const bool left = p.flags & kLeft, lower = p.flags & kLower,
             trans = p.flags & kTrans, unit = p.flags & kUnit;
  const uint k = p.k, m = p.m, n = p.n;
  const uint lines = left ? n : m;
  if (gid >= p.batch * lines) return;
  const uint b = gid / lines, line = gid % lines;
  device const float* A = a + b * k * k;
  device const float* B = bm + b * m * n;
  device float* X = x + b * m * n;
  // The system solved by this thread is E y = r with E = op(A) (left) or
  // op(A)^T (right); E is lower triangular iff lower ^ trans ^ right.
  const bool e_lower = lower != trans != !left;
  #define E(i, j) ((trans != !left) ? A[(j) * k + (i)] : A[(i) * k + (j)])
  #define R(i) (left ? B[(i) * n + line] : B[line * n + (i)])
  #define Y(i) (left ? X[(i) * n + line] : X[line * n + (i)])
  if (e_lower) {
    for (uint i = 0; i < k; ++i) {
      float s = R(i);
      for (uint j = 0; j < i; ++j) s -= E(i, j) * Y(j);
      Y(i) = unit ? s : s / E(i, i);
    }
  } else {
    for (uint ii = k; ii-- > 0;) {
      float s = R(ii);
      for (uint j = ii + 1; j < k; ++j) s -= E(ii, j) * Y(j);
      Y(ii) = unit ? s : s / E(ii, ii);
    }
  }
  #undef E
  #undef R
  #undef Y
}

// LU with partial pivoting of a column-major m x n matrix (LAPACK layout,
// lda = m), in place (a may equal lu); pivots are 0-based, permutation the
// row order they produce. One thread per matrix. Singular pivots (exactly
// zero) are left in place, as sgetrf does.
kernel void small_getrf(device const float* a [[buffer(0)]],
                        device float* lu [[buffer(1)]],
                        device int* piv [[buffer(2)]],
                        device int* perm [[buffer(3)]],
                        constant SmallParams& p [[buffer(4)]],
                        uint b [[thread_position_in_grid]]) {
  if (b >= p.batch) return;
  const uint m = p.m, n = p.n, kk = p.k;
  device const float* A = a + b * m * n;
  device float* U = lu + b * m * n;
  device int* pv = piv + b * kk;
  device int* pm = perm + b * m;
  if (A != U) for (uint i = 0; i < m * n; ++i) U[i] = A[i];
  #define M(i, j) U[(j) * m + (i)]
  for (uint j = 0; j < kk; ++j) {
    uint piv_row = j;
    float best = fabs(M(j, j));
    for (uint i = j + 1; i < m; ++i) {
      const float v = fabs(M(i, j));
      if (v > best) { best = v; piv_row = i; }
    }
    pv[j] = (int)piv_row;
    if (piv_row != j) {
      for (uint c = 0; c < n; ++c) {
        const float t = M(j, c); M(j, c) = M(piv_row, c); M(piv_row, c) = t;
      }
    }
    const float d = M(j, j);
    if (d != 0.0f) {
      for (uint i = j + 1; i < m; ++i) M(i, j) /= d;
    }
    for (uint i = j + 1; i < m; ++i) {
      const float l = M(i, j);
      if (l == 0.0f) continue;
      for (uint c = j + 1; c < n; ++c) M(i, c) -= l * M(j, c);
    }
  }
  #undef M
  for (uint i = 0; i < m; ++i) pm[i] = (int)i;
  for (uint i = 0; i < kk; ++i) {
    const int t = pm[i]; pm[i] = pm[pv[i]]; pm[pv[i]] = t;
  }
}
