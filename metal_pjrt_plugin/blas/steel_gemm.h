// Native half-precision GEMM ("steel", after the MLX kernels it is ported
// from; see steel_gemm_msl.h). MPSMatrixMultiplication has no bf16, so the MPS
// path (mps_gemm.h) stages bf16 through f32 copies; this path runs bf16/f16
// directly with simdgroup-matrix MSL kernels that accumulate in f32.
//
// Same GemmParams contract as RunMpsGemm (row-major, operand buffers already
// bound to (MTLBuffer, offset)), except that it is launched on an rt::Stream
// like any other kernel. C is read from (and written to) params.c in place
// when beta != 0; with beta == 0 it is never read.
//
// Backend policy (UseSteelGemm): f16/bf16 inputs go to steel, and so do f32
// GEMMs with a BlasLt epilogue (applied in steel's store, SteelEpilogue) up
// to batch*m*n*k = 2^33; other f32 GEMMs go to MPS (plus a second pass for
// an epilogue, metal_blas.cc). METAL_PJRT_GEMM=mps|steel forces one
// backend for every supported case (A/B testing).
// METAL_PJRT_STEEL_TILE=bm,bn,bk,wm,wn overrides the tile config.
#ifndef METAL_PJRT_PLUGIN_BLAS_STEEL_GEMM_H_
#define METAL_PJRT_PLUGIN_BLAS_STEEL_GEMM_H_

#include <string>

#include "absl/status/status.h"
#include "metal_pjrt_plugin/blas/mps_gemm.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace blas {

struct SteelTile {
  int bm = 64, bn = 64, bk = 16, wm = 1, wn = 2;
};

// cuBLASLt epilogue applied in the store (semantics in blas_lt_support.h):
// v = alpha A B + beta C + bias[col]; aux = v; D = act(v), in f32, rounded
// once. Device pointers; `bias` holds n elements and `aux` has D's layout,
// both of D's dtype.
struct SteelEpilogue {
  int act = 0;  // 0 none, 1 ReLU, 2 GELU (tanh), 3 SiLU
  const void* bias = nullptr;  // null: no bias
  void* aux = nullptr;         // null: no aux output
};

// Tile config the dispatcher picks for `p` (after any env override).
SteelTile ChooseSteelTile(const GemmParams& p);

// True when the steel kernel can run `p` (dtypes, index ranges). `why`
// receives the reason when not.
bool SteelGemmSupports(const GemmParams& p, std::string* why = nullptr);

// Policy: whether the BLAS dispatcher should use steel for `p` (dtype and
// epilogue policy plus METAL_PJRT_GEMM, and SteelGemmSupports).
bool UseSteelGemm(const GemmParams& p, bool has_epilogue = false);

// Encodes the GEMM onto `stream`. `tile` null means ChooseSteelTile(p);
// `epi` null means none. InvalidArgumentError for bad shapes/strides or
// unsupported cases; errors from kernel compilation and launch pass through.
absl::Status RunSteelGemm(rt::Device* device, rt::Stream* stream,
                          const GemmParams& p,
                          const SteelTile* tile = nullptr,
                          const SteelEpilogue* epi = nullptr);

}  // namespace blas
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_BLAS_STEEL_GEMM_H_
