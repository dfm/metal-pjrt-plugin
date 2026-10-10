// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// BLAS support for the Metal StreamExecutor: f32 GEMMs on Metal Performance
// Shaders (mps_gemm.h), f16/bf16 on the steel kernels (steel_gemm.h), or,
// with 2..8 rows (or columns) in the x W^T layout, on the wide gemv
// (gemv.h).
//
// GEMMs reach it through gpu::BlasLt (CublasLtMatmulThunk,
// "__cublas$lt$matmul"): GemmRewriter routes *every* GEMM there when the
// compute capability is OneAPI, which is what the Metal platform reports
// (gemm_rewriter.cc, GetNonFp8GemmCustomCallTarget). The legacy
// blas::BlasSupport::DoBlasGemm* entry points return Unimplemented.
//
// Supported: f32, f16, bf16 inputs; output of the same type, or f32 for
// f16/bf16 inputs; alpha/beta real; transposes; leading dims; strided batches
// (including stride-0 broadcast); every BlasLt epilogue (bias, ReLU, GELU,
// SiLU, with or without aux output). Steel applies an epilogue in its store;
// f32 GEMMs run on MPS plus a small MSL kernel over D afterwards. What is
// supported is ValidateMatmul (blas_lt_support.h), which the compiler also
// runs, so unsupported GEMMs (e.g. f16/bf16 shapes past steel's 32-bit index
// limits) are refused at compile time.
// Complex/f64/int8 and GEMV/TRSM/Scal are unimplemented.
#ifndef METAL_PJRT_BLAS_METAL_BLAS_H_
#define METAL_PJRT_BLAS_METAL_BLAS_H_

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/blas/blas_lt_support.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"
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
    // non-trivial `epilogue` runs in steel's (or the gemv's) store when
    // `epilogue_kernel` is null, else `epilogue_kernel` applies it to D
    // after an MPS GEMM (f32); see metal_blas.cc.
    MatmulPlan(metal_pjrt::rt::Device* device,
               metal_pjrt::blas::GemmParams params, bool swap_operands,
               EpilogueSpec epilogue = {},
               const metal_pjrt::rt::Kernel* epilogue_kernel = nullptr)
        : device_(device),
          params_(params),
          swap_operands_(swap_operands),
          epilogue_(epilogue),
          epilogue_kernel_(epilogue_kernel) {}

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
    const metal_pjrt::rt::Kernel* epilogue_kernel_;  // owned by the device

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
      : blas_lt_(device) {}
  ~MetalBlas() override = default;

  gpu::BlasLt* GetBlasLt() override { return &blas_lt_; }

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

 private:
  MetalBlasLt blas_lt_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_BLAS_METAL_BLAS_H_
