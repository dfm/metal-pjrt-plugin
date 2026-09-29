// Small-M half-precision GEMM ("wide gemv", after MLX's gemv_wide; see
// kernels/gemv.metal): D = x W^T for 2..8 vectors x and K >= 512, streaming
// the matrix W once per <= 5 vectors instead of through steel's (mostly
// empty) 16-row MMA tiles. Same GemmParams contract as RunSteelGemm (row-major, bf16/f16
// inputs, output of the input type or f32, the cuBLASLt epilogue applied in
// the store). XLA-free.
//
// It applies to the "NT" form, !a.transpose && b.transpose (both operands
// contiguous along K), in either orientation: m small (the vectors are A's
// rows, the matrix B's stored rows) or n small (the vectors are B's stored
// rows, the matrix A; D is written transposed through its strides). This is
// x @ W.T for a [out, in] weight, JAX's einsum("...i,oi->...o"), which is
// what MLX routes to gemv_wide too; other layouts and larger m stay on
// steel, as in MLX. m == 1 is not taken (XLA lowers matrix-vector products to
// reductions; a BlasLt one keeps its steel path).
#ifndef METAL_PJRT_BLAS_GEMV_H_
#define METAL_PJRT_BLAS_GEMV_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace blas {

// MLX's gemv_wide_config (matmul.cpp): vectors per pass (at most 5, split
// evenly over the passes), K lanes per row and the grid's x extent.
struct GemvPlan {
  bool vectors_are_b = false;  // n is the small side
  int vecs_per_tg = 0;
  int k_lanes = 32;
  int grid_x = 1;
};

// Largest vector count the wide gemv takes. MLX goes to 15 (three passes
// of five) ahead of its 64-row steel tiles; against steel's 16-row tile
// (ChooseSteelTile) the gemv wins up to 8 vectors on the LLM shapes of
// bench/gemm_bench.cc (6-40%), and loses from 12 on (to 23% for a
// 151936-row vocabulary matrix, where it is 5% behind already at 8).
inline constexpr int kGemvMaxVectors = 8;

// Smallest K the wide gemv takes. With K = 128 or 256 steel's 16-row tile
// is faster once there is enough work to fill the GPU (up to 2.8x; batched
// decode attention, 1024 x [4..8, 128] x [128, 128]^T, 1.9-2.0x), while from
// K = 512 the gemv ties or wins at every batch (docs/performance.md). MLX
// has no such limit.
inline constexpr int kGemvMinK = 512;

// The plan for `p` when the wide gemv applies (dtypes, NT layout, one side
// 2..kGemvMaxVectors, K >= kGemvMinK, K and every stride a multiple of 4
// elements and 8-byte aligned operands, 32-bit index ranges, `gpu_family`
// >= 9: MLX keeps pre-M3 GPUs on its other kernels), else nullopt.
std::optional<GemvPlan> ChooseGemv(const GemmParams& p, int gpu_family);

// The kernel for these types and plan: the MSL (kernels/gemv.metal plus one
// instantiation) and its function name; the switches are function
// constants, GemvConstants.
struct GemvKernelSource {
  std::string msl;
  std::string function;
};
GemvKernelSource GemvKernel(MpsDType in, MpsDType out, int vecs_per_tg,
                            int k_lanes);
std::vector<rt::FunctionConstant> GemvConstants(bool use_c,
                                                const SteelEpilogue& epi);

// Encodes the GEMM onto `stream` with `plan` (from ChooseGemv(p, ...)).
// `epi` null means none. InvalidArgumentError when the plan does not apply
// to `p`; kernel compilation and launch errors pass through.
absl::Status RunGemv(rt::Device* device, rt::Stream* stream,
                     const GemmParams& p, const GemvPlan& plan,
                     const SteelEpilogue* epi = nullptr);

}  // namespace blas
}  // namespace metal_pjrt

#endif  // METAL_PJRT_BLAS_GEMV_H_
