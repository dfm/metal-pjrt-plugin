// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/complex_scatter.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool MetalComplexScatterSplitter::IsSplittable(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kScatter || instr.operand_count() != 3 ||
      instr.shape().element_type() != C64 ||
      Cast<HloScatterInstruction>(&instr)->unique_indices()) {
    return false;
  }
  const HloInstruction* root = instr.to_apply()->root_instruction();
  return root->opcode() == HloOpcode::kAdd &&
         root->operand(0)->opcode() == HloOpcode::kParameter &&
         root->operand(1)->opcode() == HloOpcode::kParameter &&
         root->operand(0)->parameter_number() !=
             root->operand(1)->parameter_number();
}

absl::StatusOr<bool> MetalComplexScatterSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloComputation* f32_add = nullptr;  // shared by every split scatter
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!IsSplittable(*instr)) continue;
      auto* scatter = Cast<HloScatterInstruction>(instr);
      if (f32_add == nullptr) {
        const Shape scalar = ShapeUtil::MakeShape(F32, {});
        HloComputation::Builder b("metal_complex_scatter_add");
        HloInstruction* x = b.AddInstruction(
            HloInstruction::CreateParameter(0, scalar, "x"));
        HloInstruction* y = b.AddInstruction(
            HloInstruction::CreateParameter(1, scalar, "y"));
        b.AddInstruction(
            HloInstruction::CreateBinary(scalar, HloOpcode::kAdd, x, y));
        f32_add = module->AddEmbeddedComputation(b.Build());
      }
      auto add = [&](std::unique_ptr<HloInstruction> new_instr) {
        HloInstruction* added = comp->AddInstruction(std::move(new_instr));
        added->set_metadata(scatter->metadata());
        return added;
      };
      auto part = [&](HloInstruction* z, HloOpcode op) {
        return add(HloInstruction::CreateUnary(
            ShapeUtil::ChangeElementType(z->shape(), F32), op, z));
      };
      const Shape f32 = ShapeUtil::ChangeElementType(scatter->shape(), F32);
      HloInstruction* halves[2];
      for (int i = 0; i < 2; ++i) {
        const HloOpcode op = i == 0 ? HloOpcode::kReal : HloOpcode::kImag;
        halves[i] = add(HloInstruction::CreateScatter(
            f32, part(scatter->scatter_operands()[0], op),
            scatter->scatter_indices(),
            part(scatter->scatter_updates()[0], op), f32_add,
            scatter->scatter_dimension_numbers(),
            scatter->indices_are_sorted(), /*unique_indices=*/false));
      }
      HloInstruction* result = add(HloInstruction::CreateBinary(
          scatter->shape(), HloOpcode::kComplex, halves[0], halves[1]));
      TF_RETURN_IF_ERROR(scatter->ReplaceAllUsesWith(result));
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(scatter));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
