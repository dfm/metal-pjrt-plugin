#ifndef METAL_PJRT_PLUGIN_COMPILER_PASSES_DOT_UPCAST_H_
#define METAL_PJRT_PLUGIN_COMPILER_PASSES_DOT_UPCAST_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// True for a kDot whose operand element type is narrower than its result
// (e.g. bf16 x bf16 -> f32 from preferred_element_type, or from
// DotAlgorithmRewriter for ALG_DOT_BF16_BF16_F32*), with no algorithm or a
// bf16 one.
bool IsNarrowOperandDot(const HloInstruction* instr);

// Converts the operands of every narrow-operand kDot to the result type, and
// wraps the converts and the dot in a kLoop fusion (priority fusion never
// fuses into a kDot, so the converts would otherwise be separate kernels).
//
// Dots that GemmRewriter leaves alone (smaller than
// xla_gpu_gemm_rewrite_size_threshold, or of a type the GEMM path does not
// take) are emitted elementwise inside a loop fusion by EmitDotLoop, whose
// EmitMulAdd multiplies in the operand type and only then converts to the
// accumulator (elemental_hlo_to_mlir.cc). A bf16 x bf16 -> f32 dot therefore
// rounds every product to bf16 (rel. error ~1e-2), and an integer dot
// multiplies in the narrow type. Upcasting the operands makes the products
// exact, like the GEMM path.
//
// Must run after GemmRewriter (the end of the base
// OptimizeHloPostLayoutAssignment): earlier, it would also upcast big bf16
// dots and send them to the f32 GEMM instead of the bf16 one. Fusion runs
// after it and may fuse producers and consumers into the new fusion.
class MetalDotOperandUpcaster : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "metal-dot-operand-upcaster";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_PLUGIN_COMPILER_PASSES_DOT_UPCAST_H_
