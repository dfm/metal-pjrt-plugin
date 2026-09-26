#include "metal_pjrt_plugin/compiler/passes/dot_upcast.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool IsNarrowOperandDot(const HloInstruction* instr) {
  // An explicit algorithm (e.g. ALG_DOT_BF16_BF16_F32) fixes the operand
  // types; leave those dots alone.
  if (instr->opcode() != HloOpcode::kDot ||
      instr->precision_config().algorithm() != PrecisionConfig::ALG_UNSET) {
    return false;
  }
  PrimitiveType out = instr->shape().element_type();
  if (!primitive_util::IsFloatingPointType(out) &&
      !primitive_util::IsIntegralType(out)) {
    return false;
  }
  for (const HloInstruction* op : instr->operands()) {
    PrimitiveType in = op->shape().element_type();
    if (in != out &&
        primitive_util::BitWidth(in) < primitive_util::BitWidth(out)) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<bool> MetalDotOperandUpcaster::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* dot : comp->MakeInstructionPostOrder()) {
      if (!IsNarrowOperandDot(dot)) continue;
      PrimitiveType out = dot->shape().element_type();
      std::vector<HloInstruction*> to_fuse = {dot};
      for (int i = 0; i < dot->operand_count(); ++i) {
        HloInstruction* op = dot->mutable_operand(i);
        PrimitiveType in = op->shape().element_type();
        if (in == out ||
            primitive_util::BitWidth(in) >= primitive_util::BitWidth(out)) {
          continue;
        }
        HloInstruction* cvt = comp->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(op->shape(), out), op));
        cvt->set_metadata(dot->metadata());
        TF_RETURN_IF_ERROR(dot->ReplaceOperandWithDifferentShape(i, cvt));
        to_fuse.push_back(cvt);
      }
      // Dots are never fused (priority fusion's IsFusible), so fuse the
      // converts here; otherwise each would be its own kernel.
      comp->CreateFusionInstruction(to_fuse,
                                    HloInstruction::FusionKind::kLoop);
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
