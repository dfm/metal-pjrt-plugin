// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_COMPLEX_SCATTER_H_
#define METAL_PJRT_COMPILER_PASSES_COMPLEX_SCATTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Splits a complex64 scatter-add whose indices may repeat into two f32
// scatter-adds, one per component, joined by kComplex:
//   scatter(z, i, u, add) = complex(scatter(re z, i, re u, add),
//                                   scatter(im z, i, im u, add))
// Complex addition is componentwise, so this is exact. XLA would combine
// colliding complex64 updates with a 64-bit compare-and-swap, which Metal
// does not have; the f32 halves use Metal's 32-bit float atomics.
//
// Only single-operand scatters whose combiner is exactly an add of its two
// parameters are split. An overwrite combiner is handled in the kernel
// lowering instead (metal_pjrt/codegen: a complex overwrite is one 8-byte
// store, which a per-component split could tear). Other combiners are
// refused by CheckPostGemmRewriter.
class MetalComplexScatterSplitter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "metal-complex-scatter-splitter";
  }

  // Whether `instr` is a scatter this pass splits.
  static bool IsSplittable(const HloInstruction& instr);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_COMPLEX_SCATTER_H_
