// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/pool_rewriter.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// `comp` is exactly `op(p0, p1)` (for kCompare, with `direction`).
bool IsBinaryOfParams(const HloComputation* comp, HloOpcode op,
                      Comparison::Direction direction = {}) {
  const HloInstruction* root = comp->root_instruction();
  if (comp->num_parameters() != 2 || root->opcode() != op ||
      root->operand_count() != 2 ||
      root->operand(0) != comp->parameter_instruction(0) ||
      root->operand(1) != comp->parameter_instruction(1)) {
    return false;
  }
  return op != HloOpcode::kCompare ||
         root->comparison_direction() == direction;
}

// The window, if `sas` is a max pool's gradient the kernel computes exactly.
std::optional<std::vector<int64_t>> MatchPoolMaxBwd(
    const HloSelectAndScatterInstruction* sas) {
  const HloInstruction* x = sas->operand(0);
  const HloInstruction* dy = sas->operand(1);
  const HloInstruction* init = sas->operand(2);
  const Shape& shape = x->shape();
  const PrimitiveType type = shape.element_type();
  if (type != F32 && type != F16 && type != BF16) return std::nullopt;
  if (dy->shape().element_type() != type ||
      sas->shape().element_type() != type) {
    return std::nullopt;
  }
  for (const Shape* s : {&shape, &dy->shape(), &sas->shape()}) {
    if (s->has_layout() && !LayoutUtil::IsMonotonicWithDim0Major(s->layout())) {
      return std::nullopt;
    }
  }
  const int64_t rank = shape.dimensions().size();
  if (rank == 0) return std::nullopt;
  const int64_t elements = ShapeUtil::ElementsIn(shape);
  if (elements == 0 || elements >= (int64_t{1} << 31)) return std::nullopt;
  if (init->opcode() != HloOpcode::kConstant) return std::nullopt;
  if (!IsBinaryOfParams(sas->select(), HloOpcode::kCompare,
                        Comparison::Direction::kGe) ||
      !IsBinaryOfParams(sas->scatter(), HloOpcode::kAdd)) {
    return std::nullopt;
  }
  const Window& window = sas->window();
  if (window.dimensions_size() != rank) return std::nullopt;
  std::vector<int64_t> sizes;
  int64_t per_window = 1;
  for (int64_t d = 0; d < rank; ++d) {
    const WindowDimension& w = window.dimensions(d);
    if (w.size() < 1 || w.stride() != w.size() || w.padding_low() != 0 ||
        w.padding_high() != 0 || w.window_dilation() != 1 ||
        w.base_dilation() != 1 || w.window_reversal() ||
        dy->shape().dimensions(d) != shape.dimensions(d) / w.size()) {
      return std::nullopt;
    }
    per_window *= w.size();
    sizes.push_back(w.size());
  }
  if (per_window > 64) return std::nullopt;
  // The kernel windows at most two adjacent dims (x as [A, D1, D2, B]).
  int64_t first = -1, last = -1;
  for (int64_t d = 0; d < rank; ++d) {
    if (sizes[d] > 1) {
      if (first < 0) first = d;
      last = d;
    }
  }
  if (last - first > 1) return std::nullopt;
  return sizes;
}

}  // namespace

absl::StatusOr<bool> MetalPoolMaxBwdRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloSelectAndScatterInstruction*> candidates;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kSelectAndScatter) {
        candidates.push_back(Cast<HloSelectAndScatterInstruction>(instr));
      }
    }
    for (HloSelectAndScatterInstruction* sas : candidates) {
      std::optional<std::vector<int64_t>> window = MatchPoolMaxBwd(sas);
      if (!window.has_value()) continue;
      // The init as an f32 bit pattern (exact for f16/bf16/f32).
      absl::StatusOr<Literal> init32 =
          sas->operand(2)->literal().Convert(F32);
      if (!init32.ok()) continue;
      const float init = init32->GetFirstElement<float>();
      uint32_t init_bits;
      std::memcpy(&init_bits, &init, sizeof(init_bits));
      HloInstruction* x = sas->mutable_operand(0);
      HloInstruction* dy = sas->mutable_operand(1);
      Shape x_shape = x->shape();
      LayoutUtil::SetToDefaultLayout(&x_shape);
      Shape dy_shape = dy->shape();
      LayoutUtil::SetToDefaultLayout(&dy_shape);
      Shape result_shape = sas->shape();
      LayoutUtil::SetToDefaultLayout(&result_shape);
      HloInstruction* call = comp->AddInstruction(
          HloInstruction::CreateCustomCall(
              result_shape, {x, dy}, "metal$pool_max_bwd",
              {x_shape, dy_shape}, /*opaque=*/"",
              CustomCallApiVersion::API_VERSION_TYPED_FFI));
      GpuBackendConfig config;
      config.mutable_custom_call_backend_config()->set_attributes(
          absl::StrFormat("{window = array<i64: %s>, init = 0x%08X : f32}",
                          absl::StrJoin(*window, ", "), init_bits));
      TF_RETURN_IF_ERROR(call->set_backend_config(config));
      call->set_metadata(sas->metadata());
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(sas, call));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
