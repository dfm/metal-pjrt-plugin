#ifndef METAL_PJRT_PLUGIN_BLAS_BLAS_LT_SUPPORT_H_
#define METAL_PJRT_PLUGIN_BLAS_BLAS_LT_SUPPORT_H_

// What MetalBlasLt supports, without Metal: shared by MetalBlasLt and the
// compiler's post-GemmRewriter checks (compiler/passes/postconditions.h).

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace metal {

// BlasLt epilogues
//
// cuBLASLt semantics (what XLA's GemmRewriter fuses, see
// gemm_rewriter.cc FuseVectorBiasAdd / FuseReluActivation /
// FuseGeluActivation):
//   D   = act(alpha * op(A) op(B) + beta * C + bias)
//   aux = alpha * op(A) op(B) + beta * C + bias      (*WithAux only)
// * bias is a vector with one element per index of the output's most minor
//   physical dimension (cuBLASLt: per row of the column-major D), shared by
//   all rows and batches; its element type is D's.
// * GELU is the tanh approximation (the rewriter only matches
//   0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))).
// * SILU is x * sigmoid(x) (only fused on ROCm >= 7, supported anyway).
// * aux has D's shape, element type, leading dimension and batch stride.
//
// Steel applies the epilogue in its store, in f32 with one rounding
// (steel_gemm.h SteelEpilogue). With METAL_PJRT_GEMM=mps, one elementwise
// kernel applies it in f32 after the MPS GEMM (which leaves alpha AB + beta C
// in D, rounded) and writes D (and aux).

enum class Activation { kNone = 0, kReLU = 1, kGELU = 2, kSILU = 3 };

struct EpilogueSpec {
  bool bias = false;
  Activation act = Activation::kNone;
  bool aux = false;
  bool trivial() const { return !bias && act == Activation::kNone; }
};

absl::StatusOr<EpilogueSpec> DecodeEpilogue(gpu::BlasLt::Epilogue e);

// f32, f16 or bf16 operands of one type; output of the same type, or f32.
absl::Status CheckBlasLtTypes(xla::PrimitiveType a, xla::PrimitiveType b,
                              xla::PrimitiveType out);

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_PLUGIN_BLAS_BLAS_LT_SUPPORT_H_
