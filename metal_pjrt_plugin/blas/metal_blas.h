// BLAS support for the Metal StreamExecutor, backed by Metal Performance
// Shaders (see mps_gemm.h).
//
// Two entry points reach GEMMs in XLA:
//   * blas::BlasSupport::DoBlasGemm* (GemmThunk, "__cublas$gemm"): legacy
//     cuBLAS-style column-major calls. The GemmRewriter no longer emits
//     these; without an algorithm they carry no output type, so only f32 is
//     accepted there.
//   * gpu::BlasLt (CublasLtMatmulThunk, "__cublas$lt$matmul"): GemmRewriter
//     routes *every* GEMM here when the compute capability is OneAPI, which is
//     what the Metal platform reports (gemm_rewriter.cc,
//     GetNonFp8GemmCustomCallTarget). So MetalBlasLt is the path actually
//     exercised in practice; both share RunMpsGemm.
//
// Supported: f32, f16, bf16 inputs; output of the same type, or f32 for
// f16/bf16 inputs; alpha/beta real; transposes; leading dims; strided batches
// (including stride-0 broadcast); every BlasLt epilogue (bias, ReLU, GELU,
// SiLU, with or without aux output). f16/bf16 GEMMs with an epilogue run on
// steel, which applies it in its store; f32 ones (and METAL_PJRT_GEMM=mps)
// run on MPS plus a small MSL kernel over D afterwards.
// Complex/f64/int8 and GEMV/TRSM/Scal are unimplemented.
#ifndef METAL_PJRT_PLUGIN_BLAS_METAL_BLAS_H_
#define METAL_PJRT_PLUGIN_BLAS_METAL_BLAS_H_

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt_plugin/blas/blas_lt_support.h"
#include "metal_pjrt_plugin/blas/mps_gemm.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

// gpu::BlasLt over MPS. Plans are immutable after creation.
class MetalBlasLt : public gpu::BlasLt {
 public:
  explicit MetalBlasLt(metal_pjrt::rt::Device* device) : device_(device) {}

  class MatmulPlan : public gpu::BlasLt::MatmulPlan {
   public:
    // `params` holds everything but buffers, in row-major form; operands
    // bind from MemoryArgs with a/b swapped when `swap_operands`. A
    // non-trivial `epilogue` runs in steel's store when `epilogue_kernel`
    // is null, else `epilogue_kernel` applies it to D after an MPS GEMM
    // (METAL_PJRT_GEMM=mps); see metal_blas.cc.
    MatmulPlan(metal_pjrt::rt::Device* device,
               metal_pjrt::blas::GemmParams params, bool swap_operands,
               EpilogueSpec epilogue = {},
               std::shared_ptr<metal_pjrt::rt::Kernel> epilogue_kernel =
                   nullptr)
        : device_(device),
          params_(params),
          swap_operands_(swap_operands),
          epilogue_(epilogue),
          epilogue_kernel_(std::move(epilogue_kernel)) {}

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;
    absl::StatusOr<std::vector<gpu::BlasLt::MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;
    absl::Status SetAlgorithm(
        const gpu::BlasLt::MatmulAlgorithm& algorithm) override;

   private:
    metal_pjrt::rt::Device* device_;
    metal_pjrt::blas::GemmParams params_;
    bool swap_operands_;
    EpilogueSpec epilogue_;
    std::shared_ptr<metal_pjrt::rt::Kernel> epilogue_kernel_;

    absl::StatusOr<std::pair<const void*, void*>> EpilogueBuffers(
        const gpu::BlasLt::MemoryArgs& args) const;
    absl::Status RunSteelWithEpilogue(
        Stream* stream, const metal_pjrt::blas::GemmParams& p,
        const gpu::BlasLt::MemoryArgs& args) const;
    absl::Status RunEpilogue(Stream* stream,
                             const gpu::BlasLt::MemoryArgs& args) const;
  };

  absl::Status Init() override { return absl::OkStatus(); }
  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(
      const gpu::GemmConfig& cfg, Epilogue epilogue) const override;
  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(
      const gpu::GroupedGemmConfig& config, Epilogue epilogue) const override;

 private:
  metal_pjrt::rt::Device* device_;
};

class MetalBlas : public blas::BlasSupport {
 public:
  explicit MetalBlas(metal_pjrt::rt::Device* device)
      : device_(device), blas_lt_(device) {}
  ~MetalBlas() override = default;

  gpu::BlasLt* GetBlasLt() override { return &blas_lt_; }

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  // Column-major (BLAS convention) GEMM, possibly strided-batched.
  absl::Status DoGemm(Stream* stream, blas::Transpose transa,
                      blas::Transpose transb, uint64_t m, uint64_t n,
                      uint64_t k, blas::DataType type_ab,
                      blas::DataType type_c, const void* alpha,
                      const void* beta, const DeviceAddressBase& a, int lda,
                      int64_t stride_a, const DeviceAddressBase& b, int ldb,
                      int64_t stride_b, DeviceAddressBase* c, int ldc,
                      int64_t stride_c, int batch_count);

  metal_pjrt::rt::Device* device_;
  MetalBlasLt blas_lt_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_PLUGIN_BLAS_METAL_BLAS_H_
