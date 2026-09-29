// Dense linear algebra on the CPU (Apple Accelerate LAPACK/BLAS) for the
// Metal platform.
//
// On unified memory the fastest path for small and medium dense
// factorizations is the host LAPACK: every handler here synchronizes the
// stream (all previously enqueued GPU work has completed and its writes are
// visible), then runs LAPACK directly on the shared-storage buffers (a device
// pointer is the MTLBuffer's contents() address, so there is no copy) and
// returns. Later stream work is encoded after the handler returns, so it sees
// the results. Cholesky, triangular solve and getrf of matrices of up to 32
// rows/cols run on the GPU instead, with no synchronization (small_linalg.h).
//
// Two families of targets:
//
// * Targets of the MetalLinalgRewriter HLO pass (kCholesky/kTriangularSolve).
//   Operands and results have XLA's default row-major layout.
//     metal$cholesky(a[..., n, n]) -> l[..., n, n]; attr lower (bool).
//       The unused triangle of the result is zeroed; a batch element that is
//       not positive definite becomes all-NaN (as on JAX's CPU backend).
//     metal$triangular_solve(a[..., m, m], b[..., M, N]) -> x[..., M, N];
//       attrs left_side, lower, unit_diagonal (bool), transpose_a (i32, the
//       TriangularSolveOptions::Transpose enum; ADJOINT == TRANSPOSE for
//       real types).
//
// * Targets of the JAX lowerings in metal_pjrt_plugin/_linalg_lowerings.py.
//   These mirror jaxlib's lapack_*_ffi calls: the lowering requests
//   column-major layouts for the matrix operands/results (XLA inserts the
//   transposes), so the handlers see LAPACK's native layout. Batch dimensions
//   are the leading ones.
//     metal$lapack_getrf(a[..., m, n]) -> (lu, pivots[..., k] (0-based),
//       permutation[..., m])
//     metal$lapack_geqrf(a[..., m, n]) -> (a, taus[..., k])
//     metal$lapack_orgqr(a[..., m, n], taus[..., k]) -> q[..., m, n]
//     metal$lapack_syevd(a[..., n, n]) -> (v[..., n, n], w[..., n]);
//       attr lower (bool)
//     metal$lapack_gesdd(a[..., m, n]) -> (a (scratch), s, u, vt);
//       attr full_matrices (bool)
//     metal$lapack_gesdd_novec(a[..., m, n]) -> (a (scratch), s)
//   A failed factorization (info != 0) turns that batch element's outputs
//   into NaN, as JAX's CPU lowering does; getrf only fails on bad arguments
//   (singular matrices are not an error, as on CPU).
//
// Only f32 is supported (JAX's x64 mode is off on this backend).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/linalg/small_linalg.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

// Accelerate's classic LAPACK (LP64, 32-bit integers) and CBLAS interfaces.
// Declared here rather than through <Accelerate/Accelerate.h> to keep the
// framework headers out of the XLA build.
extern "C" {
int spotrf_(const char* uplo, const int* n, float* a, const int* lda,
            int* info);
int sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv,
            int* info);
int sgeqrf_(const int* m, const int* n, float* a, const int* lda, float* tau,
            float* work, const int* lwork, int* info);
int sorgqr_(const int* m, const int* n, const int* k, float* a, const int* lda,
            const float* tau, float* work, const int* lwork, int* info);
int ssyevd_(const char* jobz, const char* uplo, const int* n, float* a,
            const int* lda, float* w, float* work, const int* lwork, int* iwork,
            const int* liwork, int* info);
int sgesdd_(const char* jobz, const int* m, const int* n, float* a,
            const int* lda, float* s, float* u, const int* ldu, float* vt,
            const int* ldvt, float* work, const int* lwork, int* iwork,
            int* info);
void cblas_strsm(int order, int side, int uplo, int transa, int diag, int m,
                 int n, float alpha, const float* a, int lda, float* b,
                 int ldb);
}

namespace metal_pjrt {
namespace linalg {
namespace {

namespace xffi = ::xla::ffi;

// CBLAS enum values (cblas.h).
constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;
constexpr int kCblasUpper = 121;
constexpr int kCblasLower = 122;
constexpr int kCblasNonUnit = 131;
constexpr int kCblasUnit = 132;
constexpr int kCblasLeft = 141;
constexpr int kCblasRight = 142;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Waits for all work enqueued on the stream so far. The FFI handler runs on
// the host thread that enqueues the executable's thunks, so after this the
// buffers hold their final values and nothing on the GPU is using them.
// Running the LAPACK work as a stream host task instead (HostCallback) was
// measured and not faster (docs/performance.md, "Host LAPACK as a stream
// host task").
absl::Status SyncStream(stream_executor::Stream* stream) {
  absl::StatusOr<ffi::MetalContext> ctx = ffi::GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  return ctx->stream->Synchronize();
}

absl::Status CheckF32(const char* name, xla::PrimitiveType t) {
  if (t != xla::F32) {
    return absl::UnimplementedError(
        absl::StrCat(name, ": only f32 is supported on the Metal backend"));
  }
  return absl::OkStatus();
}

// (batch, rows, cols) of a buffer whose last two dims are the matrix.
struct MatrixDims {
  int64_t batch = 1;
  int64_t rows = 0;
  int64_t cols = 0;
};

absl::StatusOr<MatrixDims> GetMatrixDims(const char* name,
                                         absl::Span<const int64_t> dims) {
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ": operand must have rank >= 2"));
  }
  MatrixDims d;
  for (size_t i = 0; i + 2 < dims.size(); ++i) d.batch *= dims[i];
  d.rows = dims[dims.size() - 2];
  d.cols = dims[dims.size() - 1];
  if (d.rows > std::numeric_limits<int>::max() ||
      d.cols > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ": matrix too large for 32-bit LAPACK"));
  }
  return d;
}

void CopyIfDistinct(void* dst, const void* src, size_t bytes) {
  if (dst != src && bytes > 0) std::memmove(dst, src, bytes);
}

absl::Status LapackError(const char* name, int info) {
  return absl::InternalError(
      absl::StrCat(name, ": LAPACK reported illegal argument ", -info,
                   metal_pjrt::kReportBug));
}

// The workspace size for a float `query` result (LAPACK's lwork = -1 call),
// at least `minimum` (the documented minimum). Above 2^24 the float can round
// below the size LAPACK then requires, so round up to the next float and
// take the documented minimum too (as jaxlib does).
absl::StatusOr<int> Workspace(const char* name, float query, int64_t minimum) {
  const double w = std::max<double>(
      std::ceil(std::nextafter(query, std::numeric_limits<float>::infinity())),
      static_cast<double>(std::max<int64_t>(minimum, 1)));
  if (w > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ": workspace too large for 32-bit LAPACK"));
  }
  return static_cast<int>(w);
}

//===--------------------------------------------------------------------===//
// HLO-level ops (row-major).
//===--------------------------------------------------------------------===//

absl::Status Cholesky(stream_executor::Stream* stream, xffi::AnyBuffer a,
                      xffi::Result<xffi::AnyBuffer> out, bool lower) {
  constexpr char kName[] = "metal$cholesky";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  if (d->rows != d->cols) {
    return absl::InvalidArgumentError("metal$cholesky: matrix must be square");
  }
  absl::StatusOr<ffi::MetalContext> ctx = ffi::GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  if (UseSmallCholesky(d->batch, d->rows)) {
    return RunSmallCholesky(ctx->device, ctx->stream,
                            static_cast<const float*>(a.untyped_data()),
                            static_cast<float*>(out->untyped_data()),
                            d->batch, d->rows, lower);
  }
  if (absl::Status s = ctx->stream->Synchronize(); !s.ok()) return s;
  const int n = static_cast<int>(d->rows);
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  if (n == 0) return absl::OkStatus();
  // Row-major lower L (A = L L^T) is, read column-major, the upper factor U =
  // L^T with A^T = A = U^T U; likewise row-major upper is column-major lower.
  const char uplo = lower ? 'U' : 'L';
  for (int64_t b = 0; b < d->batch; ++b) {
    float* m = x + b * int64_t{n} * n;
    int info = 0;
    spotrf_(&uplo, &n, m, &n, &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(m, m + int64_t{n} * n, kNaN);
      continue;
    }
    // Zero the (row-major) triangle LAPACK did not write.
    for (int i = 0; i < n; ++i) {
      float* row = m + int64_t{i} * n;
      if (lower) {
        std::fill(row + i + 1, row + n, 0.0f);
      } else {
        std::fill(row, row + i, 0.0f);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status TriangularSolve(stream_executor::Stream* stream,
                             xffi::AnyBuffer a, xffi::AnyBuffer b,
                             xffi::Result<xffi::AnyBuffer> out, bool left_side,
                             bool lower, bool unit_diagonal,
                             int32_t transpose_a) {
  constexpr char kName[] = "metal$triangular_solve";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  if (absl::Status s = CheckF32(kName, b.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> da = GetMatrixDims(kName, a.dimensions());
  if (!da.ok()) return da.status();
  absl::StatusOr<MatrixDims> db = GetMatrixDims(kName, b.dimensions());
  if (!db.ok()) return db.status();
  const int64_t k = left_side ? db->rows : db->cols;
  if (da->rows != da->cols || da->rows != k || da->batch != db->batch) {
    return absl::InvalidArgumentError(
        "metal$triangular_solve: incompatible operand shapes");
  }
  // TriangularSolveOptions::Transpose: 1 = NO_TRANSPOSE, 2 = TRANSPOSE,
  // 3 = ADJOINT (the same as TRANSPOSE for real types).
  const bool trans = transpose_a == 2 || transpose_a == 3;
  absl::StatusOr<ffi::MetalContext> ctx = ffi::GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  if (UseSmallTrsm(db->batch, db->rows, db->cols, left_side)) {
    return RunSmallTrsm(ctx->device, ctx->stream,
                        static_cast<const float*>(a.untyped_data()),
                        static_cast<const float*>(b.untyped_data()),
                        static_cast<float*>(out->untyped_data()), db->batch,
                        db->rows, db->cols, left_side, lower, trans,
                        unit_diagonal);
  }
  if (absl::Status s = ctx->stream->Synchronize(); !s.ok()) return s;
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, b.untyped_data(), b.size_bytes());
  const int m = static_cast<int>(db->rows);
  const int n = static_cast<int>(db->cols);
  if (m == 0 || n == 0) return absl::OkStatus();
  const float* av = static_cast<const float*>(a.untyped_data());
  for (int64_t i = 0; i < db->batch; ++i) {
    cblas_strsm(kCblasRowMajor, left_side ? kCblasLeft : kCblasRight,
                lower ? kCblasLower : kCblasUpper,
                trans ? kCblasTrans : kCblasNoTrans,
                unit_diagonal ? kCblasUnit : kCblasNonUnit, m, n, 1.0f,
                av + i * k * k, static_cast<int>(k), x + i * int64_t{m} * n, n);
  }
  return absl::OkStatus();
}

//===--------------------------------------------------------------------===//
// JAX-level ops (column-major matrices).
//===--------------------------------------------------------------------===//

absl::Status Getrf(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> lu,
                   xffi::Result<xffi::AnyBuffer> pivots,
                   xffi::Result<xffi::AnyBuffer> permutation) {
  constexpr char kName[] = "metal$lapack_getrf";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  if (pivots->element_type() != xla::S32 ||
      permutation->element_type() != xla::S32) {
    return absl::InvalidArgumentError(
        "metal$lapack_getrf: pivots/permutation must be s32");
  }
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  absl::StatusOr<ffi::MetalContext> ctx = ffi::GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  if (UseSmallGetrf(d->batch, d->rows, d->cols)) {
    return RunSmallGetrf(ctx->device, ctx->stream,
                         static_cast<const float*>(a.untyped_data()),
                         static_cast<float*>(lu->untyped_data()),
                         static_cast<int32_t*>(pivots->untyped_data()),
                         static_cast<int32_t*>(permutation->untyped_data()),
                         d->batch, d->rows, d->cols);
  }
  if (absl::Status s = ctx->stream->Synchronize(); !s.ok()) return s;
  float* x = static_cast<float*>(lu->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  const int m = static_cast<int>(d->rows);
  const int n = static_cast<int>(d->cols);
  const int k = std::min(m, n);
  int32_t* piv = static_cast<int32_t*>(pivots->untyped_data());
  int32_t* perm = static_cast<int32_t*>(permutation->untyped_data());
  for (int64_t b = 0; b < d->batch; ++b) {
    float* mb = x + b * int64_t{m} * n;
    int32_t* pb = piv + b * k;
    int32_t* qb = perm + b * m;
    int info = 0;
    if (m > 0 && n > 0) {
      const int lda = std::max(m, 1);
      sgetrf_(&m, &n, mb, &lda, pb, &info);
    }
    if (info < 0) return LapackError(kName, info);
    for (int i = 0; i < m; ++i) qb[i] = i;
    for (int i = 0; i < k; ++i) {
      pb[i] -= 1;  // 1-based to 0-based
      std::swap(qb[i], qb[pb[i]]);
    }
  }
  return absl::OkStatus();
}

absl::Status Geqrf(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> out,
                   xffi::Result<xffi::AnyBuffer> taus) {
  constexpr char kName[] = "metal$lapack_geqrf";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  const int m = static_cast<int>(d->rows);
  const int n = static_cast<int>(d->cols);
  const int k = std::min(m, n);
  if (k == 0) return absl::OkStatus();
  float* t = static_cast<float*>(taus->untyped_data());
  const int lda = std::max(m, 1);
  int info = 0;
  int lwork = -1;
  float wq = 0;
  sgeqrf_(&m, &n, x, &lda, t, &wq, &lwork, &info);
  if (info < 0) return LapackError(kName, info);
  absl::StatusOr<int> ws = Workspace(kName, wq, n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < d->batch; ++b) {
    sgeqrf_(&m, &n, x + b * int64_t{m} * n, &lda, t + b * k, work.data(),
            &lwork, &info);
    if (info < 0) return LapackError(kName, info);
  }
  return absl::OkStatus();
}

absl::Status Orgqr(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::AnyBuffer taus, xffi::Result<xffi::AnyBuffer> out) {
  constexpr char kName[] = "metal$lapack_orgqr";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  absl::Span<const int64_t> tdims = taus.dimensions();
  const int k = tdims.empty() ? 0 : static_cast<int>(tdims.back());
  const int m = static_cast<int>(d->rows);
  const int n = static_cast<int>(d->cols);
  if (!(m >= n && n >= k)) {
    return absl::InvalidArgumentError(
        "metal$lapack_orgqr: requires m >= n >= k");
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  if (n == 0) return absl::OkStatus();
  const float* t = static_cast<const float*>(taus.untyped_data());
  const int lda = std::max(m, 1);
  int info = 0;
  int lwork = -1;
  float wq = 0;
  sorgqr_(&m, &n, &k, x, &lda, t, &wq, &lwork, &info);
  if (info < 0) return LapackError(kName, info);
  absl::StatusOr<int> ws = Workspace(kName, wq, n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < d->batch; ++b) {
    sorgqr_(&m, &n, &k, x + b * int64_t{m} * n, &lda, t + b * k, work.data(),
            &lwork, &info);
    if (info < 0) return LapackError(kName, info);
  }
  return absl::OkStatus();
}

absl::Status Syevd(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> v,
                   xffi::Result<xffi::AnyBuffer> w, bool lower) {
  constexpr char kName[] = "metal$lapack_syevd";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  if (d->rows != d->cols) {
    return absl::InvalidArgumentError("metal$lapack_syevd: matrix must be square");
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(v->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  const int n = static_cast<int>(d->rows);
  if (n == 0) return absl::OkStatus();
  float* wv = static_cast<float*>(w->untyped_data());
  const char jobz = 'V';
  const char uplo = lower ? 'L' : 'U';
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  float wq = 0;
  int iwq = 0;
  ssyevd_(&jobz, &uplo, &n, x, &n, wv, &wq, &lwork, &iwq, &liwork, &info);
  if (info < 0) return LapackError(kName, info);
  // Documented minimums for jobz = 'V'.
  absl::StatusOr<int> ws =
      Workspace(kName, wq, 1 + 6 * int64_t{n} + 2 * int64_t{n} * n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  liwork = std::max(3 + 5 * n, iwq);
  std::vector<float> work(lwork);
  std::vector<int> iwork(liwork);
  for (int64_t b = 0; b < d->batch; ++b) {
    float* mb = x + b * int64_t{n} * n;
    float* wb = wv + b * n;
    ssyevd_(&jobz, &uplo, &n, mb, &n, wb, work.data(), &lwork, iwork.data(),
            &liwork, &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(mb, mb + int64_t{n} * n, kNaN);
      std::fill(wb, wb + n, kNaN);
    }
  }
  return absl::OkStatus();
}

absl::Status GesddImpl(stream_executor::Stream* stream, xffi::AnyBuffer a,
                       float* x, float* s, float* u, float* vt,
                       bool full_matrices) {
  constexpr char kName[] = "metal$lapack_gesdd";
  if (absl::Status st = CheckF32(kName, a.element_type()); !st.ok()) return st;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  if (absl::Status st = SyncStream(stream); !st.ok()) return st;
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  const int m = static_cast<int>(d->rows);
  const int n = static_cast<int>(d->cols);
  const int k = std::min(m, n);
  if (k == 0) return absl::OkStatus();
  const bool uv = u != nullptr;
  const char jobz = !uv ? 'N' : (full_matrices ? 'A' : 'S');
  const int ucols = full_matrices ? m : k;
  const int vtrows = full_matrices ? n : k;
  const int ldu = std::max(m, 1);
  const int ldvt = uv ? std::max(vtrows, 1) : 1;
  float udummy = 0, vtdummy = 0;
  int info = 0;
  int lwork = -1;
  float wq = 0;
  std::vector<int> iwork(8 * static_cast<size_t>(k));
  sgesdd_(&jobz, &m, &n, x, &m, s, uv ? u : &udummy, &ldu,
          uv ? vt : &vtdummy, &ldvt, &wq, &lwork, iwork.data(), &info);
  if (info < 0) return LapackError(kName, info);
  // Documented minimums: jobz = 'N' needs 3mn + max(mx, 7mn); 'S' and 'A'
  // need 4mn^2 + 7mn and 4mn^2 + 6mn + mx (LAPACK 3.7+), older releases
  // 3mn + max(mx, 4mn^2 + 4mn). Take the largest that applies.
  const int64_t mn = k;
  const int64_t mx = std::max(m, n);
  const int64_t min_work =
      !uv ? 3 * mn + std::max(mx, 7 * mn)
          : std::max({4 * mn * mn + 7 * mn, 4 * mn * mn + 6 * mn + mx,
                      3 * mn + std::max(mx, 4 * mn * mn + 4 * mn)});
  absl::StatusOr<int> ws = Workspace(kName, wq, min_work);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < d->batch; ++b) {
    float* xb = x + b * int64_t{m} * n;
    float* sb = s + b * k;
    float* ub = uv ? u + b * int64_t{m} * ucols : &udummy;
    float* vb = uv ? vt + b * int64_t{vtrows} * n : &vtdummy;
    sgesdd_(&jobz, &m, &n, xb, &m, sb, ub, &ldu, vb, &ldvt, work.data(),
            &lwork, iwork.data(), &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(sb, sb + k, kNaN);
      if (uv) {
        std::fill(ub, ub + int64_t{m} * ucols, kNaN);
        std::fill(vb, vb + int64_t{vtrows} * n, kNaN);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status Gesdd(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> x,
                   xffi::Result<xffi::AnyBuffer> s,
                   xffi::Result<xffi::AnyBuffer> u,
                   xffi::Result<xffi::AnyBuffer> vt, bool full_matrices) {
  return GesddImpl(stream, a, static_cast<float*>(x->untyped_data()),
                   static_cast<float*>(s->untyped_data()),
                   static_cast<float*>(u->untyped_data()),
                   static_cast<float*>(vt->untyped_data()), full_matrices);
}

absl::Status GesddNoVec(stream_executor::Stream* stream, xffi::AnyBuffer a,
                        xffi::Result<xffi::AnyBuffer> x,
                        xffi::Result<xffi::AnyBuffer> s) {
  return GesddImpl(stream, a, static_cast<float*>(x->untyped_data()),
                   static_cast<float*>(s->untyped_data()), nullptr, nullptr,
                   /*full_matrices=*/false);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalCholesky, Cholesky,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<bool>("lower"));
XLA_FFI_DEFINE_HANDLER(kMetalTriangularSolve, TriangularSolve,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<bool>("left_side")
                           .Attr<bool>("lower")
                           .Attr<bool>("unit_diagonal")
                           .Attr<int32_t>("transpose_a"));
XLA_FFI_DEFINE_HANDLER(kMetalGetrf, Getrf,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER(kMetalGeqrf, Geqrf,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER(kMetalOrgqr, Orgqr,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER(kMetalSyevd, Syevd,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<bool>("lower"));
XLA_FFI_DEFINE_HANDLER(kMetalGesdd, Gesdd,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<bool>("full_matrices"));
XLA_FFI_DEFINE_HANDLER(kMetalGesddNoVec, GesddNoVec,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$cholesky", "METAL",
                         kMetalCholesky);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$triangular_solve",
                         "METAL", kMetalTriangularSolve);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_getrf", "METAL",
                         kMetalGetrf);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_geqrf", "METAL",
                         kMetalGeqrf);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_orgqr", "METAL",
                         kMetalOrgqr);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_syevd", "METAL",
                         kMetalSyevd);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_gesdd", "METAL",
                         kMetalGesdd);
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_gesdd_novec",
                         "METAL", kMetalGesddNoVec);

}  // namespace linalg
}  // namespace metal_pjrt
