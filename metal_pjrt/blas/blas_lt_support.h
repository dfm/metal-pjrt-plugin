#ifndef METAL_PJRT_BLAS_BLAS_LT_SUPPORT_H_
#define METAL_PJRT_BLAS_BLAS_LT_SUPPORT_H_

// What MetalBlasLt supports, without Metal: shared by MetalBlasLt and the
// compiler's post-GemmRewriter checks (compiler/passes/postconditions.h).

#include <cstdint>

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
// Steel (f16/bf16) applies the epilogue in its store, in f32 with one
// rounding (steel_gemm.h SteelEpilogue). For f32, one elementwise kernel
// applies it after the MPS GEMM (which leaves alpha AB + beta C in D) and
// writes D (and aux).

enum class Activation { kNone = 0, kReLU = 1, kGELU = 2, kSILU = 3 };

struct EpilogueSpec {
  bool bias = false;
  Activation act = Activation::kNone;
  bool aux = false;
  bool trivial() const { return !bias && act == Activation::kNone; }
};

// A GEMM MetalBlasLt can run, in the row-major form MPS and steel take.
struct ValidatedMatmul {
  // A column-major output is computed as D^T = B^T A^T: lhs and rhs are
  // swapped and every layout (c and out included) is viewed transposed.
  bool swap = false;
  gpu::MatrixLayout lhs, rhs, c, out;  // lhs/rhs batch_size = batch
  int64_t m = 0, n = 0, k = 0, batch = 0;
  EpilogueSpec epi;
};

// Everything MetalBlasLt requires of a GEMM, without Metal: a supported
// epilogue; f32, f16 or bf16 operands of one type with an output of that
// type or f32; real alpha; no transposed layouts; C laid out like D when
// beta != 0; and the index limits of the kernel that will run it (steel's
// 32-bit index math for f16/bf16; the f32 epilogue kernel's uint32 grid).
// CheckPostGemmRewriter calls it at compile time and GetMatmulPlan again as
// a backstop, so a GEMM that compiles gets a plan.
absl::StatusOr<ValidatedMatmul> ValidateMatmul(const gpu::GemmConfig& cfg,
                                               gpu::BlasLt::Epilogue epilogue);

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_BLAS_BLAS_LT_SUPPORT_H_
