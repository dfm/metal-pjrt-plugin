#include "metal_pjrt_plugin/blas/metal_blas.h"

#include <algorithm>
#include <any>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt_plugin/blas/mps_gemm.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace metal {

namespace rt = ::metal_pjrt::rt;
namespace mps = ::metal_pjrt::blas;

namespace {

bool Unsupported(const char* what) {
  LOG(ERROR) << "Metal BLAS: " << what << " is not implemented";
  return false;
}

absl::StatusOr<mps::MpsDType> FromBlasType(blas::DataType t) {
  switch (t) {
    case blas::DataType::kFloat:
      return mps::MpsDType::kF32;
    case blas::DataType::kHalf:
      return mps::MpsDType::kF16;
    case blas::DataType::kBF16:
      return mps::MpsDType::kBF16;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Metal BLAS: unsupported data type ", blas::DataTypeString(t)));
  }
}

absl::StatusOr<mps::MpsDType> FromPrimitiveType(xla::PrimitiveType t) {
  switch (t) {
    case xla::F32:
      return mps::MpsDType::kF32;
    case xla::F16:
      return mps::MpsDType::kF16;
    case xla::BF16:
      return mps::MpsDType::kBF16;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Metal BLAS: unsupported element type ",
          xla::PrimitiveType_Name(t)));
  }
}

// Output types the MPS path supports for a given input type.
bool SupportedTypes(mps::MpsDType in, mps::MpsDType out) {
  return in == out || out == mps::MpsDType::kF32;
}

// Resolves a raw device pointer into (MTLBuffer, byte offset).
absl::Status Bind(rt::Device* device, const DeviceAddressBase& mem,
                  absl::string_view name, mps::MpsOperand* op) {
  if (mem.opaque() == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal BLAS: operand ", name, " is null"));
  }
  absl::StatusOr<rt::BufferRef> ref = device->Resolve(mem.opaque());
  if (!ref.ok()) {
    return absl::Status(ref.status().code(),
                        absl::StrCat("Metal BLAS: operand ", name, ": ",
                                     ref.status().message()));
  }
  op->buffer = static_cast<void*>(ref->buffer);
  op->offset = ref->offset;
  return absl::OkStatus();
}

rt::Stream* RtStream(Stream* stream) {
  // MetalStream::platform_specific_handle() exposes its rt::Stream. Going
  // through the handle keeps this library independent of metal_executor.
  return static_cast<rt::Stream*>(stream->platform_specific_handle().stream);
}

absl::Status Encode(rt::Device* device, Stream* stream,
                    const mps::GemmParams& params) {
  rt::Stream* rs = RtStream(stream);
  if (rs == nullptr) {
    return absl::InvalidArgumentError(
        "Metal BLAS: stream has no Metal handle (not a Metal stream?)");
  }
  VLOG(3) << "Metal BLAS: " << mps::GemmParamsDebugString(params);
  return rs->EncodeExternal([&](void* cmd) {
    return mps::RunMpsGemm(static_cast<void*>(device->mtl()), cmd, params);
  });
}

// alpha/beta arrive as float for f16/bf16/f32 (the typed wrappers upcast
// half-precision scales), double for f64.
double ReadScale(const void* p, blas::DataType scale_type) {
  if (scale_type == blas::DataType::kDouble) {
    return *static_cast<const double*>(p);
  }
  return *static_cast<const float*>(p);
}

}  // namespace

// ---------------------------------------------------------------------------
// Legacy BlasSupport GEMMs (column-major, cuBLAS convention)

absl::Status MetalBlas::DoGemm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, blas::DataType type_ab,
    blas::DataType type_c, const void* alpha, const void* beta,
    const DeviceAddressBase& a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, int ldb, int64_t stride_b,
    DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count) {
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType in, FromBlasType(type_ab));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType out, FromBlasType(type_c));
  if (!SupportedTypes(in, out)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal BLAS: unsupported type combination ",
                     blas::DataTypeString(type_ab), " -> ",
                     blas::DataTypeString(type_c)));
  }
  mps::GemmParams p;
  p.m = static_cast<int64_t>(m);
  p.n = static_cast<int64_t>(n);
  p.k = static_cast<int64_t>(k);
  p.batch_count = batch_count;
  p.alpha = ReadScale(alpha, type_c);
  p.beta = ReadScale(beta, type_c);
  // Column-major description; conjugate transpose == transpose for reals.
  p.a = {nullptr, 0, lda, stride_a, in, transa != blas::Transpose::kNoTranspose};
  p.b = {nullptr, 0, ldb, stride_b, in, transb != blas::Transpose::kNoTranspose};
  p.c = {nullptr, 0, ldc, stride_c, out, false};
  ABSL_RETURN_IF_ERROR(Bind(device_, a, "A", &p.a));
  ABSL_RETURN_IF_ERROR(Bind(device_, b, "B", &p.b));
  ABSL_RETURN_IF_ERROR(Bind(device_, *c, "C", &p.c));
  mps::ColumnMajorToRowMajor(&p);
  return Encode(device_, stream, p);
}

absl::Status MetalBlas::DoBlasGemm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
    const void* alpha, const DeviceAddressBase& a, int lda,
    const DeviceAddressBase& b, int ldb, const void* beta,
    DeviceAddressBase* c, int ldc, const EngineOptions& engine_options,
    blas::CallContext context) {
  return DoBlasGemmStridedBatched(stream, transa, transb, m, n, k, dtype,
                                  alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0,
                                  1, engine_options, context);
}

absl::Status MetalBlas::DoBlasGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
    const void* alpha, const DeviceAddressBase& a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, int ldb, int64_t stride_b, const void* beta,
    DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count,
    const EngineOptions& engine_options, blas::CallContext context) {
  // This entry point only carries the input type. XLA's RunGemm also routes
  // (bf16|f16) x (bf16|f16) -> f32 here, so infer an f32 output from C's
  // size: a half-precision C is exactly 2 bytes per element and can never be
  // large enough to hold the f32 extent.
  blas::DataType type_c = dtype;
  if (dtype == blas::DataType::kHalf || dtype == blas::DataType::kBF16) {
    const uint64_t extent =
        static_cast<uint64_t>(batch_count > 0 ? batch_count - 1 : 0) *
            static_cast<uint64_t>(stride_c) +
        (n > 0 ? (n - 1) * static_cast<uint64_t>(ldc) : 0) + m;
    if (c->size() >= 4 * extent) type_c = blas::DataType::kFloat;
  }
  return DoGemm(stream, transa, transb, m, n, k, dtype, type_c, alpha, beta, a,
                lda, stride_a, b, ldb, stride_b, c, ldc, stride_c,
                batch_count);
}

absl::Status MetalBlas::DoBlasGemmWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, const void* alpha,
    const DeviceAddressBase& a, blas::DataType type_a, int lda,
    const DeviceAddressBase& b, blas::DataType type_b, int ldb,
    const void* beta, DeviceAddressBase* c, blas::DataType type_c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext context) {
  return DoBlasGemmStridedBatchedWithAlgorithm(
      stream, transa, transb, m, n, k, alpha, a, type_a, lda, 0, b, type_b,
      ldb, 0, beta, c, type_c, ldc, 0, 1, computation_type, algorithm,
      engine_options, output_profile_result, context);
}

absl::Status MetalBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, const void* alpha,
    const DeviceAddressBase& a, blas::DataType type_a, int lda,
    int64_t stride_a, const DeviceAddressBase& b, blas::DataType type_b,
    int ldb, int64_t stride_b, const void* beta, DeviceAddressBase* c,
    blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    const EngineOptions& engine_options,
    blas::ProfileResult* output_profile_result, blas::CallContext context) {
  if (type_a != type_b) {
    return absl::UnimplementedError(
        absl::StrCat("Metal BLAS: mixed input types ",
                     blas::DataTypeString(type_a), " x ",
                     blas::DataTypeString(type_b)));
  }
  // MPS picks its own accumulation precision (f32 for f16/bf16 inputs as far
  // as we can tell); computation_type and algorithm are accepted and ignored.
  auto start = std::chrono::steady_clock::now();
  if (output_profile_result != nullptr) {
    ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    start = std::chrono::steady_clock::now();
  }
  absl::Status s =
      DoGemm(stream, transa, transb, m, n, k, type_a, type_c, alpha, beta, a,
             lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, batch_count);
  if (output_profile_result != nullptr) {
    if (s.ok()) s = stream->BlockHostUntilDone();
    output_profile_result->set_is_valid(s.ok());
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count());
  }
  return s;
}

bool MetalBlas::GetBlasGemmAlgorithms(
    Stream* stream, const gpu::MatrixDescriptor& a,
    const gpu::MatrixDescriptor& b, gpu::OutputMatrixDescriptor* c,
    const void* alpha, const void* beta,
    std::vector<blas::AlgorithmType>* out_algorithms) {
  out_algorithms->clear();
  out_algorithms->push_back(blas::kDefaultGemmAlgo);
  return true;
}

absl::StatusOr<bool> MetalBlas::IsMainStreamSet() const { return true; }

absl::Status MetalBlas::GetVersion(std::string* version) {
  *version = "MetalPerformanceShaders MPSMatrixMultiplication";
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// BlasLt

absl::StatusOr<gpu::BlasLt::MatmulPlanPtr> MetalBlasLt::GetMatmulPlan(
    const gpu::GemmConfig& cfg, Epilogue epilogue) const {
  if (epilogue != Epilogue::kDefault) {
    return absl::UnimplementedError(
        absl::StrCat("Metal BlasLt: epilogue ", static_cast<int>(epilogue),
                     " is not supported"));
  }
  if (cfg.alpha.imag() != 0.0) {
    return absl::UnimplementedError("Metal BlasLt: complex alpha");
  }
  gpu::MatrixLayout lhs = cfg.lhs_layout, rhs = cfg.rhs_layout,
                    c = cfg.c_layout, out = cfg.output_layout;
  for (const gpu::MatrixLayout* l : {&lhs, &rhs, &c, &out}) {
    if (l->transpose != blas::Transpose::kNoTranspose) {
      return absl::UnimplementedError(
          absl::StrCat("Metal BlasLt: transposed layout ", l->ToString()));
    }
  }
  // As in cuBLASLt: an operand without a batch dimension is broadcast
  // (its batch_stride is 0).
  const int64_t batch = std::max(lhs.batch_size, rhs.batch_size);
  lhs.batch_size = rhs.batch_size = batch;
  if (out.batch_size != batch) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal BlasLt: batch mismatch ", out.ToString()));
  }

  // MPS writes a row-major result. For a column-major output use
  //   D^T = (A B)^T = B^T A^T,
  // i.e. swap the operands and view every layout transposed (the inverse of
  // gpu::MakeOutputColumnMajor).
  const bool swap = out.order == gpu::MatrixLayout::Order::kColumnMajor;
  if (swap) {
    std::swap(lhs, rhs);
    lhs.Transpose();
    rhs.Transpose();
    c.Transpose();
    out.Transpose();
  }

  const int64_t m = out.num_rows, n = out.num_cols, k = lhs.num_cols;
  if (lhs.num_rows != m || rhs.num_rows != k || rhs.num_cols != n) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal BlasLt: inconsistent shapes lhs=", lhs.ToString(),
        " rhs=", rhs.ToString(), " out=", out.ToString()));
  }
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType ta, FromPrimitiveType(lhs.dtype));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType tb, FromPrimitiveType(rhs.dtype));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType tout, FromPrimitiveType(out.dtype));
  if (ta != tb || !SupportedTypes(ta, tout)) {
    return absl::UnimplementedError(absl::StrCat(
        "Metal BlasLt: unsupported types ", xla::PrimitiveType_Name(lhs.dtype),
        " x ", xla::PrimitiveType_Name(rhs.dtype), " -> ",
        xla::PrimitiveType_Name(out.dtype)));
  }
  if (cfg.beta != 0.0 &&
      (c.dtype != out.dtype || c.order != out.order ||
       c.leading_dim_stride != out.leading_dim_stride ||
       c.batch_stride != out.batch_stride || c.num_rows != out.num_rows ||
       c.num_cols != out.num_cols)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal BlasLt: C layout ", c.ToString(),
                     " differs from output layout ", out.ToString()));
  }

  // A row-major logical matrix is its stored matrix; a column-major one is the
  // transpose of the row-major view of its storage.
  auto operand = [](const gpu::MatrixLayout& l, mps::MpsDType t) {
    mps::MpsOperand o;
    o.ld = l.leading_dim_stride;
    o.batch_stride = l.batch_size > 1 ? l.batch_stride : 0;
    o.dtype = t;
    o.transpose = l.order == gpu::MatrixLayout::Order::kColumnMajor;
    return o;
  };
  mps::GemmParams p;
  p.m = m;
  p.n = n;
  p.k = k;
  p.batch_count = batch;
  p.alpha = cfg.alpha.real();
  p.beta = cfg.beta;
  p.a = operand(lhs, ta);
  p.b = operand(rhs, tb);
  p.c = operand(out, tout);  // row-major after the swap above
  return std::make_unique<MatmulPlan>(device_, p, swap);
}

absl::StatusOr<gpu::BlasLt::MatmulPlanPtr> MetalBlasLt::GetMatmulPlan(
    const gpu::GroupedGemmConfig& config, Epilogue epilogue) const {
  return absl::UnimplementedError("Metal BlasLt: grouped GEMM");
}

absl::Status MetalBlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {
  mps::GemmParams p = params_;
  const DeviceAddressBase& a = swap_operands_ ? args.b : args.a;
  const DeviceAddressBase& b = swap_operands_ ? args.a : args.b;
  DeviceAddressBase d = args.d;
  // D = alpha A B + beta C: MPS accumulates in place, so seed D with C when
  // they are distinct buffers (layouts were checked equal at plan creation).
  if (p.beta != 0.0 && args.c.opaque() != nullptr &&
      args.c.opaque() != d.opaque()) {
    ABSL_RETURN_IF_ERROR(stream->MemcpyD2D(&d, args.c, d.size()));
  }
  ABSL_RETURN_IF_ERROR(Bind(device_, a, "A", &p.a));
  ABSL_RETURN_IF_ERROR(Bind(device_, b, "B", &p.b));
  ABSL_RETURN_IF_ERROR(Bind(device_, d, "D", &p.c));

  auto start = std::chrono::steady_clock::now();
  if (profile_result != nullptr) {
    ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    start = std::chrono::steady_clock::now();
  }
  absl::Status s = Encode(device_, stream, p);
  if (profile_result != nullptr) {
    if (s.ok()) s = stream->BlockHostUntilDone();
    profile_result->set_is_valid(s.ok());
    profile_result->set_algorithm(0);
    profile_result->set_elapsed_time_in_ms(
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count());
  }
  return s;
}

absl::StatusOr<std::vector<gpu::BlasLt::MatmulAlgorithm>>
MetalBlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
                                       size_t max_workspace_size) const {
  // One algorithm (MPS chooses internally), no workspace.
  return std::vector<gpu::BlasLt::MatmulAlgorithm>{
      {std::any(int64_t{0}), /*workspace_size=*/0}};
}

absl::Status MetalBlasLt::MatmulPlan::SetAlgorithm(
    const gpu::BlasLt::MatmulAlgorithm& algorithm) {
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Everything else is unimplemented (not reached by XLA's GPU GEMM paths).

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, float alpha, DeviceAddress<float>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, double alpha, DeviceAddress<double>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, float alpha, DeviceAddress<std::complex<float>>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, double alpha, DeviceAddress<std::complex<double>>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, std::complex<float> alpha, DeviceAddress<std::complex<float>>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasScal(Stream* stream, uint64_t elem_count, std::complex<double> alpha, DeviceAddress<std::complex<double>>* x, int incx) {
  return Unsupported("DoBlasScal");
}

bool MetalBlas::DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n, float alpha, const DeviceAddress<float>& a, int lda, const DeviceAddress<float>& x, int incx, float beta, DeviceAddress<float>* y, int incy) {
  return Unsupported("DoBlasGemv");
}

bool MetalBlas::DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n, double alpha, const DeviceAddress<double>& a, int lda, const DeviceAddress<double>& x, int incx, double beta, DeviceAddress<double>* y, int incy) {
  return Unsupported("DoBlasGemv");
}

bool MetalBlas::DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n, std::complex<float> alpha, const DeviceAddress<std::complex<float>>& a, int lda, const DeviceAddress<std::complex<float>>& x, int incx, std::complex<float> beta, DeviceAddress<std::complex<float>>* y, int incy) {
  return Unsupported("DoBlasGemv");
}

bool MetalBlas::DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n, std::complex<double> alpha, const DeviceAddress<std::complex<double>>& a, int lda, const DeviceAddress<std::complex<double>>& x, int incx, std::complex<double> beta, DeviceAddress<std::complex<double>>* y, int incy) {
  return Unsupported("DoBlasGemv");
}

bool MetalBlas::DoBlasGemmBatched( Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<Eigen::half> a, int lda, DeviceAddressSlice<Eigen::half> b, int ldb, float beta, DeviceAddressSlice<Eigen::half> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasGemmBatched( Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<Eigen::bfloat16> a, int lda, DeviceAddressSlice<Eigen::bfloat16> b, int ldb, float beta, DeviceAddressSlice<Eigen::bfloat16> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasGemmBatched(Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, float alpha, DeviceAddressSlice<float> a, int lda, DeviceAddressSlice<float> b, int ldb, float beta, DeviceAddressSlice<float> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasGemmBatched( Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, double alpha, DeviceAddressSlice<double> a, int lda, DeviceAddressSlice<double> b, int ldb, double beta, DeviceAddressSlice<double> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasGemmBatched( Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, std::complex<float> alpha, DeviceAddressSlice<std::complex<float>> a, int lda, DeviceAddressSlice<std::complex<float>> b, int ldb, std::complex<float> beta, DeviceAddressSlice<std::complex<float>> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasGemmBatched( Stream* stream, blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, std::complex<double> alpha, DeviceAddressSlice<std::complex<double>> a, int lda, DeviceAddressSlice<std::complex<double>> b, int ldb, std::complex<double> beta, DeviceAddressSlice<std::complex<double>> c, int ldc, int batch_count, const EngineOptions& engine_options, ScratchAllocator* scratch_allocator, blas::CallContext context) {
  return Unsupported("DoBlasGemmBatched");
}

bool MetalBlas::DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, float alpha, const DeviceAddress<float>& a, int lda, DeviceAddress<float>* b, int ldb) {
  return Unsupported("DoBlasTrsm");
}

bool MetalBlas::DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, double alpha, const DeviceAddress<double>& a, int lda, DeviceAddress<double>* b, int ldb) {
  return Unsupported("DoBlasTrsm");
}

bool MetalBlas::DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, std::complex<float> alpha, const DeviceAddress<std::complex<float>>& a, int lda, DeviceAddress<std::complex<float>>* b, int ldb) {
  return Unsupported("DoBlasTrsm");
}

bool MetalBlas::DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, std::complex<double> alpha, const DeviceAddress<std::complex<double>>& a, int lda, DeviceAddress<std::complex<double>>* b, int ldb) {
  return Unsupported("DoBlasTrsm");
}

bool MetalBlas::DoBlasTrsmBatched( Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, float alpha, const DeviceAddress<float*>& as, int lda, DeviceAddress<float*>* bs, int ldb, int batch_count) {
  return Unsupported("DoBlasTrsmBatched");
}

bool MetalBlas::DoBlasTrsmBatched( Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, double alpha, const DeviceAddress<double*>& as, int lda, DeviceAddress<double*>* bs, int ldb, int batch_count) {
  return Unsupported("DoBlasTrsmBatched");
}

bool MetalBlas::DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, std::complex<float> alpha, const DeviceAddress<std::complex<float>*>& as, int lda, DeviceAddress<std::complex<float>*>* bs, int ldb, int batch_count) {
  return Unsupported("DoBlasTrsmBatched");
}

bool MetalBlas::DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo, blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n, std::complex<double> alpha, const DeviceAddress<std::complex<double>*>& as, int lda, DeviceAddress<std::complex<double>*>* bs, int ldb, int batch_count) {
  return Unsupported("DoBlasTrsmBatched");
}

}  // namespace metal
}  // namespace stream_executor
