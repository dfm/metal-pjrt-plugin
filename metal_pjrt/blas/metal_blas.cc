#include "metal_pjrt/blas/metal_blas.h"

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
#include "metal_pjrt/blas/blas_lt_support.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"  // steel GEMM dispatch
#include "metal_pjrt/runtime/metal_runtime.h"
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
  // f16/bf16 run on the steel kernels (steel_gemm.h), f32 on MPS.
  if (params.a.dtype != mps::MpsDType::kF32) {
    return mps::RunSteelGemm(device, rs, params);
  }
  return rs->EncodeExternal(
      [&](void* cmd) {
        return mps::RunMpsGemm(device, cmd, params);
      },
      mps::GemmWork(params));
}

// BlasLt epilogues: semantics and DecodeEpilogue in blas_lt_support.h.

// Must match the `EpiParams` struct in the generated MSL.
struct EpilogueParams {
  uint32_t m;             // rows (row-major view)
  uint32_t n;             // columns == bias length
  uint64_t ld;            // elements between rows of D / aux
  uint64_t batch_stride;  // elements between batches of D / aux
};

constexpr char kEpilogueKernelName[] = "xla_metal_gemm_epilogue";

// The f32 epilogue kernel run after an MPS GEMM.
std::string EpilogueSource(const EpilogueSpec& e) {
  std::string act;
  switch (e.act) {
    case Activation::kNone:
      act = "  return x;\n";
      break;
    case Activation::kReLU:
      // max() would drop NaN; cuBLAS/XLA's max propagates it.
      act = "  return isnan(x) ? x : (x > 0.0f ? x : 0.0f);\n";
      break;
    case Activation::kGELU:
      act =
          "  const float k = 0.7978845608028654f;  // sqrt(2/pi)\n"
          "  return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x "
          "* x * x)));\n";
      break;
    case Activation::kSILU:
      act = "  return x / (1.0f + precise::exp(-x));\n";
      break;
  }
  return absl::StrCat(
      "#include <metal_stdlib>\nusing namespace metal;\n",
      "inline float act_(float x) {\n", act, "}\n",
      "struct EpiParams { uint m; uint n; ulong ld; ulong batch_stride; };\n",
      "kernel void ", kEpilogueKernelName, "(\n",
      "    device float* d [[buffer(0)]],\n",
      "    device const float* bias [[buffer(1)]],\n",
      "    device float* aux [[buffer(2)]],\n",
      "    constant EpiParams& p [[buffer(3)]],\n",
      "    uint3 gid [[thread_position_in_grid]]) {\n",
      "  if (gid.x >= p.n || gid.y >= p.m) return;\n",
      "  ulong i = ulong(gid.z) * p.batch_stride + ulong(gid.y) * p.ld + "
      "gid.x;\n",
      "  float x = d[i];\n",
      e.bias ? "  x += bias[gid.x];\n" : "",
      e.aux ? "  aux[i] = x;\n" : "",
      "  d[i] = act_(x);\n}\n");
}

}  // namespace

// ---------------------------------------------------------------------------
// Legacy BlasSupport GEMMs (GemmThunk, "__cublas$gemm"): the GemmRewriter
// sends every GEMM to BlasLt on OneAPI (tripwired), and hlo_checks refuses
// any other GEMM custom call at compile time.

absl::Status MetalBlas::DoBlasGemm(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
    const void* alpha, const DeviceAddressBase& a, int lda,
    const DeviceAddressBase& b, int ldb, const void* beta,
    DeviceAddressBase* c, int ldc, const EngineOptions& engine_options,
    blas::CallContext context) {
  return absl::UnimplementedError("Metal BLAS: legacy GEMM (use BlasLt)");
}

absl::Status MetalBlas::DoBlasGemmStridedBatched(
    Stream* stream, blas::Transpose transa, blas::Transpose transb,
    uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
    const void* alpha, const DeviceAddressBase& a, int lda, int64_t stride_a,
    const DeviceAddressBase& b, int ldb, int64_t stride_b, const void* beta,
    DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count,
    const EngineOptions& engine_options, blas::CallContext context) {
  return absl::UnimplementedError("Metal BLAS: legacy GEMM (use BlasLt)");
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
  return absl::UnimplementedError("Metal BLAS: legacy GEMM (use BlasLt)");
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
  return absl::UnimplementedError("Metal BLAS: legacy GEMM (use BlasLt)");
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
  ABSL_ASSIGN_OR_RETURN(ValidatedMatmul v, ValidateMatmul(cfg, epilogue));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType ta, FromPrimitiveType(v.lhs.dtype));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType tb, FromPrimitiveType(v.rhs.dtype));
  ABSL_ASSIGN_OR_RETURN(mps::MpsDType tout, FromPrimitiveType(v.out.dtype));

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
  p.m = v.m;
  p.n = v.n;
  p.k = v.k;
  p.batch_count = v.batch;
  p.alpha = cfg.alpha.real();
  p.beta = cfg.beta;
  p.a = operand(v.lhs, ta);
  p.b = operand(v.rhs, tb);
  p.c = operand(v.out, tout);  // row-major after the swap
  // f16/bf16 run only on steel, which applies any epilogue in its store
  // (bias index = column of the row-major view of D, i.e. the stored minor
  // dimension).
  if (ta != mps::MpsDType::kF32) {
    return std::make_unique<MatmulPlan>(device_, p, v.swap, v.epi);
  }
  if (v.epi.trivial()) return std::make_unique<MatmulPlan>(device_, p, v.swap);

  // MPS (f32): a second kernel over the row-major view of D: the bias index
  // is the column, aux shares D's layout.
  ABSL_ASSIGN_OR_RETURN(
      const rt::Kernel* kernel,
      device_->GetKernel(EpilogueSource(v.epi), kEpilogueKernelName));
  return std::make_unique<MatmulPlan>(device_, p, v.swap, v.epi, kernel);
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
  absl::Status s;
  if (epilogue_.trivial()) {
    s = Encode(device_, stream, p);
  } else if (epilogue_kernel_ == nullptr) {
    s = RunSteelWithEpilogue(stream, p, args);
  } else {
    s = Encode(device_, stream, p);
    if (s.ok()) s = RunEpilogue(stream, args);
  }
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

// Checks D, bias and aux against the plan and returns (bias, aux) device
// pointers, D's where the epilogue has none.
absl::StatusOr<std::pair<const void*, void*>>
MetalBlasLt::MatmulPlan::EpilogueBuffers(
    const gpu::BlasLt::MemoryArgs& args) const {
  const mps::GemmParams& p = params_;
  const uint64_t elem = mps::MpsDTypeSize(p.c.dtype);
  const uint64_t batch_stride = p.batch_count > 1 ? p.c.batch_stride : 0;
  // Extent of D (and aux) in elements.
  const uint64_t extent = (p.batch_count - 1) * batch_stride +
                          (p.m - 1) * static_cast<uint64_t>(p.c.ld) + p.n;
  if (args.d.size() < extent * elem) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal BlasLt epilogue: D has ", args.d.size(), " bytes, need ",
        extent * elem));
  }
  const void* bias = args.d.opaque();
  if (epilogue_.bias) {
    if (args.bias.opaque() == nullptr ||
        args.bias.size() < static_cast<uint64_t>(p.n) * elem) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Metal BlasLt epilogue: bias buffer (", args.bias.size(),
          " bytes) must hold ", p.n, " elements of ",
          mps::MpsDTypeName(p.c.dtype)));
    }
    bias = args.bias.opaque();
  }
  void* aux = args.d.opaque();
  if (epilogue_.aux) {
    if (args.aux.opaque() == nullptr || args.aux.size() < extent * elem) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Metal BlasLt epilogue: aux buffer (", args.aux.size(),
          " bytes) smaller than the output (", extent * elem, " bytes)"));
    }
    aux = args.aux.opaque();
  }
  return std::make_pair(bias, aux);
}

absl::Status MetalBlasLt::MatmulPlan::RunSteelWithEpilogue(
    Stream* stream, const mps::GemmParams& p,
    const gpu::BlasLt::MemoryArgs& args) const {
  if (p.m == 0 || p.n == 0 || p.batch_count == 0) return absl::OkStatus();
  ABSL_ASSIGN_OR_RETURN(auto buffers, EpilogueBuffers(args));
  rt::Stream* rs = RtStream(stream);
  if (rs == nullptr) {
    return absl::InvalidArgumentError(
        "Metal BlasLt: stream has no Metal handle (not a Metal stream?)");
  }
  mps::SteelEpilogue e;
  e.act = static_cast<int>(epilogue_.act);
  if (epilogue_.bias) e.bias = buffers.first;
  if (epilogue_.aux) e.aux = buffers.second;
  VLOG(3) << "Metal BLAS: steel + epilogue: " << mps::GemmParamsDebugString(p);
  return mps::RunSteelGemm(device_, rs, p, /*tile=*/nullptr, &e);
}

absl::Status MetalBlasLt::MatmulPlan::RunEpilogue(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args) const {
  const mps::GemmParams& p = params_;
  if (p.m == 0 || p.n == 0 || p.batch_count == 0) return absl::OkStatus();
  ABSL_ASSIGN_OR_RETURN(auto buffers, EpilogueBuffers(args));
  const void* bias = buffers.first;
  void* aux = buffers.second;
  const uint64_t batch_stride = p.batch_count > 1 ? p.c.batch_stride : 0;
  rt::Stream* rs = RtStream(stream);
  if (rs == nullptr) {
    return absl::InvalidArgumentError(
        "Metal BlasLt: stream has no Metal handle (not a Metal stream?)");
  }
  EpilogueParams ep{static_cast<uint32_t>(p.m), static_cast<uint32_t>(p.n),
                    static_cast<uint64_t>(p.c.ld), batch_stride};
  constexpr uint32_t kTx = 32, kTy = 8;
  rt::Dim3 groups{static_cast<uint32_t>((p.n + kTx - 1) / kTx),
                  static_cast<uint32_t>((p.m + kTy - 1) / kTy),
                  static_cast<uint32_t>(p.batch_count)};
  return rs->Launch(*epilogue_kernel_, groups, rt::Dim3{kTx, kTy, 1},
                    {rt::KernelArg::Buffer(args.d.opaque()),
                     rt::KernelArg::Buffer(bias), rt::KernelArg::Buffer(aux),
                     rt::KernelArg::Bytes(&ep, sizeof(ep))});
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
