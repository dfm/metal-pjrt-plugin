// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_COMPLEX_DOT_H_
#define METAL_PJRT_COMPILER_PASSES_COMPLEX_DOT_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Expands every complex64 kDot into real f32 dots:
//   real = Ar.Br - Ai.Bi,  imag = Ar.Bi + Ai.Br
// (four products, not the three of Gauss's trick, whose cancellation costs
// accuracy; cuBLAS's and XLA:CPU's complex GEMMs round each product like
// this). The real dots keep the dimension numbers (batch dimensions
// included) and the precision config, so they take the same GEMM / steel /
// loop-emitter paths as any f32 dot. A real operand (a mixed dot) is
// converted to f32 and contributes only its nonzero products.
//
// MetalBlasLt has no complex GEMM, so this must run before GemmRewriter.
// MetalCompiler runs it at the start of RunHloPasses and again after
// TriangularSolveExpander, which (like CholeskyExpander, QrExpander and
// RaggedDotRewriter before it) can emit complex dots of its own.
// complex128 dots are left alone; CheckPostGemmRewriter refuses them with
// the rest of the f64 arithmetic.
class MetalComplexDotExpander : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "metal-complex-dot-expander";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_COMPLEX_DOT_H_
