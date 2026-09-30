// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/dot_upcast.h"

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
  // An explicit algorithm fixes the operand types; leave those dots alone,
  // except the bf16 ones: DotAlgorithmRewriter turns ALG_DOT_BF16_BF16_F32*
  // into bf16 x bf16 -> f32 dots whose products must be exact in f32, and
  // bf16 -> f32 is exact.
  if (instr->opcode() != HloOpcode::kDot) return false;
  switch (instr->precision_config().algorithm()) {
    case PrecisionConfig::ALG_UNSET:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      break;
    default:
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
