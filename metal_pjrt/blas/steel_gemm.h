// Native half-precision GEMM ("steel", after the MLX kernels it is ported
// from; see steel_gemm_msl.h): bf16/f16 with simdgroup-matrix MSL kernels
// that accumulate in f32 (MPSMatrixMultiplication has no bf16).
//
// Same GemmParams contract as RunMpsGemm (row-major, operand buffers already
// bound to (MTLBuffer, offset)), except that it is launched on an rt::Stream
// like any other kernel. C is read from (and written to) params.c in place
// when beta != 0; with beta == 0 it is never read.
//
// Backend policy (metal_blas.cc): f16/bf16 GEMMs run only here (a BlasLt
// epilogue is applied in its store, SteelEpilogue); f32 goes to MPS (plus a
// second pass for an epilogue).
#ifndef METAL_PJRT_BLAS_STEEL_GEMM_H_
#define METAL_PJRT_BLAS_STEEL_GEMM_H_

#include <string>

#include "absl/status/status.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"

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

// Tile config the dispatcher picks for `p`.
SteelTile ChooseSteelTile(const GemmParams& p);

// True when the steel kernel can run `p` (dtypes, index ranges; the same
// limits as ValidateMatmul in blas_lt_support.h, which refuses the rest at
// compile time). `why` receives the reason when not.
bool SteelGemmSupports(const GemmParams& p, std::string* why = nullptr);

// Encodes the GEMM onto `stream`. `tile` null means ChooseSteelTile(p);
// `epi` null means none. InvalidArgumentError for bad shapes/strides or
// unsupported cases; errors from kernel compilation and launch pass through.
absl::Status RunSteelGemm(rt::Device* device, rt::Stream* stream,
                          const GemmParams& p,
                          const SteelTile* tile = nullptr,
                          const SteelEpilogue* epi = nullptr);

}  // namespace blas
}  // namespace metal_pjrt

#endif  // METAL_PJRT_BLAS_STEEL_GEMM_H_
