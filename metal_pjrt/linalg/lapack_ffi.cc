// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Dense linear algebra on the CPU (Apple Accelerate LAPACK/BLAS) for the
// Metal platform.
//
// On unified memory the fastest path for small and medium dense
// factorizations is the host LAPACK: every handler here synchronizes the
// stream (all previously enqueued GPU work has completed and its writes are
// visible), then runs LAPACK directly on the shared-storage buffers (a device
// pointer is the MTLBuffer's contents() address, so there is no copy) and
// returns (the LAPACK calls are in lapack_host.h). Later stream work is
// encoded after the handler returns, so it sees the results. Cholesky,
// triangular solve and getrf of matrices of up to 32 rows/cols run on the
// GPU instead, with no synchronization (small_linalg.h).
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
//     metal$lapack_gtsv(dl[..., n], d[..., n], du[..., n], b[..., n, k])
//       -> x[..., n, k]; a singular system makes that x NaN
//   A failed factorization (info != 0) turns that batch element's outputs
//   into NaN, as JAX's CPU lowering does; getrf only fails on bad arguments
//   (singular matrices are not an error, as on CPU).
//
// Only f32 is supported (JAX's x64 mode is off on this backend).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/linalg/lapack_host.h"
#include "metal_pjrt/linalg/small_linalg.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/xla_data.pb.h"

namespace metal_pjrt {
namespace linalg {
namespace {

namespace xffi = ::xla::ffi;

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

// A result buffer must have the type and element count the handler writes:
// the handlers write through raw pointers (and copy the operand into the
// first result), so a mismatch would be an out-of-bounds write.
absl::Status CheckResult(const char* name, const char* what,
                         const xffi::AnyBuffer& r, xla::PrimitiveType type,
                         int64_t elements) {
  if (r.element_type() != type ||
      static_cast<int64_t>(r.element_count()) != elements) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, ": result ", what, " has ", r.element_count(), " elements of ",
        xla::PrimitiveType_Name(r.element_type()), ", expected ", elements,
        " of ", xla::PrimitiveType_Name(type)));
  }
  return absl::OkStatus();
}

void CopyIfDistinct(void* dst, const void* src, size_t bytes) {
  if (dst != src && bytes > 0) std::memmove(dst, src, bytes);
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
  if (absl::Status s = CheckResult(kName, "l", *out, xla::F32,
                                   a.element_count());
      !s.ok()) {
    return s;
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
  return HostCholesky(x, d->batch, n, lower);
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
  if (absl::Status s = CheckResult(kName, "x", *out, xla::F32,
                                   b.element_count());
      !s.ok()) {
    return s;
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
  HostTriangularSolve(static_cast<const float*>(a.untyped_data()), x,
                      db->batch, static_cast<int>(db->rows),
                      static_cast<int>(db->cols), left_side, lower, trans,
                      unit_diagonal);
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
  const int64_t k = std::min(d->rows, d->cols);
  for (absl::Status s :
       {CheckResult(kName, "lu", *lu, xla::F32, a.element_count()),
        CheckResult(kName, "pivots", *pivots, xla::S32, d->batch * k),
        CheckResult(kName, "permutation", *permutation, xla::S32,
                    d->batch * d->rows)}) {
    if (!s.ok()) return s;
  }
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
  return HostGetrf(x, static_cast<int32_t*>(pivots->untyped_data()),
                   static_cast<int32_t*>(permutation->untyped_data()),
                   d->batch, static_cast<int>(d->rows),
                   static_cast<int>(d->cols));
}

absl::Status Geqrf(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> out,
                   xffi::Result<xffi::AnyBuffer> taus) {
  constexpr char kName[] = "metal$lapack_geqrf";
  if (absl::Status s = CheckF32(kName, a.element_type()); !s.ok()) return s;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  for (absl::Status s :
       {CheckResult(kName, "a", *out, xla::F32, a.element_count()),
        CheckResult(kName, "taus", *taus, xla::F32,
                    d->batch * std::min(d->rows, d->cols))}) {
    if (!s.ok()) return s;
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  return HostGeqrf(x, static_cast<float*>(taus->untyped_data()), d->batch,
                   static_cast<int>(d->rows), static_cast<int>(d->cols));
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
  if (absl::Status s = CheckF32(kName, taus.element_type()); !s.ok()) return s;
  if (static_cast<int64_t>(taus.element_count()) != d->batch * k) {
    return absl::InvalidArgumentError(
        "metal$lapack_orgqr: taus must have batch * k elements");
  }
  if (absl::Status s = CheckResult(kName, "q", *out, xla::F32,
                                   a.element_count());
      !s.ok()) {
    return s;
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(out->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  return HostOrgqr(x, static_cast<const float*>(taus.untyped_data()),
                   d->batch, m, n, k);
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
  for (absl::Status s :
       {CheckResult(kName, "v", *v, xla::F32, a.element_count()),
        CheckResult(kName, "w", *w, xla::F32, d->batch * d->rows)}) {
    if (!s.ok()) return s;
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* x = static_cast<float*>(v->untyped_data());
  CopyIfDistinct(x, a.untyped_data(), a.size_bytes());
  return HostSyevd(x, static_cast<float*>(w->untyped_data()), d->batch,
                   static_cast<int>(d->rows), lower);
}

// u and vt are null for gesdd_novec.
absl::Status GesddImpl(stream_executor::Stream* stream, xffi::AnyBuffer a,
                       xffi::AnyBuffer x, xffi::AnyBuffer s,
                       const xffi::AnyBuffer* u, const xffi::AnyBuffer* vt,
                       bool full_matrices) {
  constexpr char kName[] = "metal$lapack_gesdd";
  if (absl::Status st = CheckF32(kName, a.element_type()); !st.ok()) return st;
  absl::StatusOr<MatrixDims> d = GetMatrixDims(kName, a.dimensions());
  if (!d.ok()) return d.status();
  const int64_t m = d->rows, n = d->cols, k = std::min(m, n);
  for (absl::Status st :
       {CheckResult(kName, "a", x, xla::F32, a.element_count()),
        CheckResult(kName, "s", s, xla::F32, d->batch * k)}) {
    if (!st.ok()) return st;
  }
  if (u != nullptr) {
    const int64_t ucols = full_matrices ? m : k;
    const int64_t vtrows = full_matrices ? n : k;
    for (absl::Status st :
         {CheckResult(kName, "u", *u, xla::F32, d->batch * m * ucols),
          CheckResult(kName, "vt", *vt, xla::F32, d->batch * vtrows * n)}) {
      if (!st.ok()) return st;
    }
  }
  if (absl::Status st = SyncStream(stream); !st.ok()) return st;
  float* xp = static_cast<float*>(x.untyped_data());
  CopyIfDistinct(xp, a.untyped_data(), a.size_bytes());
  return HostGesdd(
      xp, static_cast<float*>(s.untyped_data()),
      u != nullptr ? static_cast<float*>(u->untyped_data()) : nullptr,
      vt != nullptr ? static_cast<float*>(vt->untyped_data()) : nullptr,
      d->batch, static_cast<int>(m), static_cast<int>(n), full_matrices);
}

absl::Status Gesdd(stream_executor::Stream* stream, xffi::AnyBuffer a,
                   xffi::Result<xffi::AnyBuffer> x,
                   xffi::Result<xffi::AnyBuffer> s,
                   xffi::Result<xffi::AnyBuffer> u,
                   xffi::Result<xffi::AnyBuffer> vt, bool full_matrices) {
  xffi::AnyBuffer ub = *u, vtb = *vt;
  return GesddImpl(stream, a, *x, *s, &ub, &vtb, full_matrices);
}

absl::Status GesddNoVec(stream_executor::Stream* stream, xffi::AnyBuffer a,
                        xffi::Result<xffi::AnyBuffer> x,
                        xffi::Result<xffi::AnyBuffer> s) {
  return GesddImpl(stream, a, *x, *s, nullptr, nullptr,
                   /*full_matrices=*/false);
}

absl::Status Gtsv(stream_executor::Stream* stream, xffi::AnyBuffer dl,
                  xffi::AnyBuffer d, xffi::AnyBuffer du, xffi::AnyBuffer b,
                  xffi::Result<xffi::AnyBuffer> x) {
  constexpr char kName[] = "metal$lapack_gtsv";
  for (const xffi::AnyBuffer* t : {&dl, &d, &du, &b}) {
    if (absl::Status s = CheckF32(kName, t->element_type()); !s.ok()) return s;
  }
  absl::StatusOr<MatrixDims> db = GetMatrixDims(kName, b.dimensions());
  if (!db.ok()) return db.status();
  for (const xffi::AnyBuffer* t : {&dl, &d, &du}) {
    if (static_cast<int64_t>(t->element_count()) != db->batch * db->rows) {
      return absl::InvalidArgumentError(
          "metal$lapack_gtsv: diagonals must have batch * n elements");
    }
  }
  if (absl::Status s = CheckResult(kName, "x", *x, xla::F32,
                                   b.element_count());
      !s.ok()) {
    return s;
  }
  if (absl::Status s = SyncStream(stream); !s.ok()) return s;
  float* xp = static_cast<float*>(x->untyped_data());
  CopyIfDistinct(xp, b.untyped_data(), b.size_bytes());
  return HostGtsv(static_cast<const float*>(dl.untyped_data()),
                  static_cast<const float*>(d.untyped_data()),
                  static_cast<const float*>(du.untyped_data()), xp, db->batch,
                  static_cast<int>(db->rows), static_cast<int>(db->cols));
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

XLA_FFI_DEFINE_HANDLER(kMetalGtsv, Gtsv,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
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
XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$lapack_gtsv", "METAL",
                         kMetalGtsv);

}  // namespace linalg
}  // namespace metal_pjrt
