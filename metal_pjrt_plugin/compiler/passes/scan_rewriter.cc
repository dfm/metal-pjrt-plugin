#include "metal_pjrt_plugin/compiler/passes/scan_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct ScanMatch {
  std::string op;
  bool reverse = false;
  int64_t n = 0;
};

// Name and identity of a binary reducer computation, if it is one we support.
std::optional<std::pair<std::string, Literal>> ReducerOp(
    const HloComputation* comp, PrimitiveType type) {
  const HloInstruction* root = comp->root_instruction();
  if (comp->num_parameters() != 2 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return std::nullopt;
  }
  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return std::make_pair(std::string("add"), LiteralUtil::Zero(type));
    case HloOpcode::kMultiply:
      return std::make_pair(std::string("mul"), LiteralUtil::One(type));
    case HloOpcode::kMaximum:
      return std::make_pair(std::string("max"), LiteralUtil::MinValue(type));
    case HloOpcode::kMinimum:
      return std::make_pair(std::string("min"), LiteralUtil::MaxValue(type));
    default:
      return std::nullopt;
  }
}

std::optional<ScanMatch> MatchScan(const HloInstruction* rw) {
  if (rw->opcode() != HloOpcode::kReduceWindow || rw->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* x = rw->operand(0);
  const HloInstruction* init = rw->operand(1);
  const Shape& shape = x->shape();
  PrimitiveType type = shape.element_type();
  if (type != F32 && type != F16 && type != BF16 && type != S32) {
    return std::nullopt;
  }
  if (!shape.IsArray() || !ShapeUtil::Equal(
                              ShapeUtil::MakeShape(type, shape.dimensions()),
                              ShapeUtil::MakeShape(
                                  rw->shape().element_type(),
                                  rw->shape().dimensions()))) {
    return std::nullopt;
  }
  for (const Shape* s : {&shape, &rw->shape()}) {
    if (s->has_layout() && !LayoutUtil::IsMonotonicWithDim0Major(s->layout())) {
      return std::nullopt;
    }
  }
  const int64_t rank = shape.dimensions().size();
  if (rank == 0 || ShapeUtil::ElementsIn(shape) == 0) return std::nullopt;
  const int64_t n = shape.dimensions(rank - 1);
  if (n < MetalScanRewriter::kMinRowLength) return std::nullopt;
  // One threadgroup per row: with few long rows the kernel would run on a
  // handful of cores, where XLA's multi-kernel scan is faster.
  const int64_t rows = ShapeUtil::ElementsIn(shape) / n;
  if (rows < MetalScanRewriter::kMinRowsForLongRows &&
      n > MetalScanRewriter::kLongRowLength) {
    return std::nullopt;
  }
  const Window& window = rw->window();
  if (window.dimensions_size() != rank) return std::nullopt;
  bool reverse = false;
  for (int64_t d = 0; d < rank; ++d) {
    const WindowDimension& w = window.dimensions(d);
    if (w.stride() != 1 || w.window_dilation() != 1 ||
        w.base_dilation() != 1 || w.window_reversal()) {
      return std::nullopt;
    }
    if (d < rank - 1) {
      if (w.size() != 1 || w.padding_low() != 0 || w.padding_high() != 0) {
        return std::nullopt;
      }
      continue;
    }
    if (w.size() != n) return std::nullopt;
    if (w.padding_low() == n - 1 && w.padding_high() == 0) {
      reverse = false;
    } else if (w.padding_low() == 0 && w.padding_high() == n - 1) {
      reverse = true;
    } else {
      return std::nullopt;
    }
  }
  if (init->opcode() != HloOpcode::kConstant) return std::nullopt;
  auto reducer = ReducerOp(rw->to_apply(), type);
  if (!reducer.has_value() || init->literal() != reducer->second) {
    return std::nullopt;
  }
  return ScanMatch{reducer->first, reverse, n};
}

}  // namespace

absl::StatusOr<bool> MetalScanRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> candidates;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kReduceWindow) {
        candidates.push_back(instr);
      }
    }
    for (HloInstruction* rw : candidates) {
      std::optional<ScanMatch> m = MatchScan(rw);
      if (!m.has_value()) continue;
      HloInstruction* x = rw->mutable_operand(0);
      Shape operand_shape = x->shape();
      LayoutUtil::SetToDefaultLayout(&operand_shape);
      Shape result_shape = rw->shape();
      LayoutUtil::SetToDefaultLayout(&result_shape);
      HloInstruction* call = comp->AddInstruction(
          HloInstruction::CreateCustomCall(
              result_shape, {x}, "metal$scan", {operand_shape},
              /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
      GpuBackendConfig config;
      config.mutable_custom_call_backend_config()->set_attributes(
          absl::StrCat("{op = \"", m->op, "\", reverse = ",
                       m->reverse ? "true" : "false", ", row_length = ", m->n,
                       " : i64}"));
      TF_RETURN_IF_ERROR(call->set_backend_config(config));
      call->set_metadata(rw->metadata());
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(rw, call));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
