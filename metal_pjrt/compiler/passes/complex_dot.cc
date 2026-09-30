// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/complex_dot.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> MetalComplexDotExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* dot : comp->MakeInstructionPostOrder()) {
      if (dot->opcode() != HloOpcode::kDot ||
          dot->shape().element_type() != C64) {
        continue;
      }
      auto add = [&](std::unique_ptr<HloInstruction> instr) {
        HloInstruction* added = comp->AddInstruction(std::move(instr));
        added->set_metadata(dot->metadata());
        return added;
      };
      // {real, imaginary} parts of each operand as f32; no imaginary part
      // (nullptr) for a real operand.
      HloInstruction* parts[2][2];
      for (int i = 0; i < 2; ++i) {
        HloInstruction* op = dot->mutable_operand(i);
        Shape f32 = ShapeUtil::ChangeElementType(op->shape(), F32);
        if (op->shape().element_type() == C64) {
          parts[i][0] =
              add(HloInstruction::CreateUnary(f32, HloOpcode::kReal, op));
          parts[i][1] =
              add(HloInstruction::CreateUnary(f32, HloOpcode::kImag, op));
        } else {
          parts[i][0] = op->shape().element_type() == F32
                            ? op
                            : add(HloInstruction::CreateConvert(f32, op));
          parts[i][1] = nullptr;
        }
      }
      const Shape f32 = ShapeUtil::ChangeElementType(dot->shape(), F32);
      auto product = [&](int a, int b) -> HloInstruction* {
        if (parts[0][a] == nullptr || parts[1][b] == nullptr) return nullptr;
        return add(HloInstruction::CreateDot(f32, parts[0][a], parts[1][b],
                                             dot->dot_dimension_numbers(),
                                             dot->precision_config()));
      };
      // The real parts always exist, so re has a term; im has none when
      // both operands are real (preferred_element_type=complex64).
      auto sum = [&](HloOpcode op, HloInstruction* x, HloInstruction* y) {
        if (x == nullptr) return y;
        if (y == nullptr) return x;
        return add(HloInstruction::CreateBinary(f32, op, x, y));
      };
      HloInstruction* re =
          sum(HloOpcode::kSubtract, product(0, 0), product(1, 1));
      HloInstruction* im = sum(HloOpcode::kAdd, product(0, 1), product(1, 0));
      if (im == nullptr) {
        im = add(HloInstruction::CreateBroadcast(
            f32,
            add(HloInstruction::CreateConstant(LiteralUtil::Zero(F32))), {}));
      }
      HloInstruction* result = add(HloInstruction::CreateBinary(
          dot->shape(), HloOpcode::kComplex, re, im));
      TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWith(result));
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(dot));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
