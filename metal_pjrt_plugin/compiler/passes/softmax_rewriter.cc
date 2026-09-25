#include "metal_pjrt_plugin/compiler/passes/softmax_rewriter.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

bool IsSupportedType(PrimitiveType t) {
  return t == F32 || t == F16 || t == BF16;
}

// Skips float-to-float converts.
HloInstruction* StripConverts(HloInstruction* h,
                              std::vector<HloInstruction*>* matched) {
  while (h->opcode() == HloOpcode::kConvert &&
         primitive_util::IsFloatingPointType(h->shape().element_type()) &&
         primitive_util::IsFloatingPointType(
             h->operand(0)->shape().element_type())) {
    matched->push_back(h);
    h = h->mutable_operand(0);
  }
  return h;
}

bool IsDefaultLayoutArray(const Shape& s) {
  return s.IsArray() && s.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(s.layout());
}

bool ReducerIs(const HloComputation* comp, HloOpcode opcode) {
  const HloInstruction* root = comp->root_instruction();
  return comp->num_parameters() == 2 && root->opcode() == opcode &&
         root->operand(0)->opcode() == HloOpcode::kParameter &&
         root->operand(1)->opcode() == HloOpcode::kParameter &&
         root->operand(0) != root->operand(1);
}

// reduce(operand, init) over the last dimension of `operand` with reducer
// `opcode` and init equal to `identity`.
bool IsMinorReduce(const HloInstruction* r, HloOpcode opcode,
                   const Literal& identity) {
  if (r->opcode() != HloOpcode::kReduce || r->operand_count() != 2) {
    return false;
  }
  const HloInstruction* in = r->operand(0);
  const HloInstruction* init = r->operand(1);
  int64_t rank = in->shape().dimensions().size();
  if (rank == 0 || r->dimensions().size() != 1 ||
      r->dimensions(0) != rank - 1) {
    return false;
  }
  if (init->opcode() != HloOpcode::kConstant ||
      init->shape().element_type() != in->shape().element_type()) {
    return false;
  }
  Literal id = identity.Convert(init->shape().element_type()).value();
  return init->literal() == id && ReducerIs(r->to_apply(), opcode);
}

// broadcast of a row-reduced value back onto `full` (dims {0..rank-2}).
bool IsRowBroadcast(const HloInstruction* b, const Shape& full) {
  if (b->opcode() != HloOpcode::kBroadcast) return false;
  int64_t rank = full.dimensions().size();
  std::vector<int64_t> dims(rank - 1);
  std::iota(dims.begin(), dims.end(), 0);
  return ShapeUtil::SameDimensions(b->shape(), full) &&
         absl::c_equal(b->dimensions(), dims);
}

struct Match {
  HloInstruction* x = nullptr;
  bool log = false;
  std::vector<HloInstruction*> intermediates;
};

// Matches D = subtract(X, broadcast(cvt(reduce_max(X)))). Returns X.
HloInstruction* MatchShifted(HloInstruction* d, Match* m) {
  if (d->opcode() != HloOpcode::kSubtract) return nullptr;
  HloInstruction* x = d->mutable_operand(0);
  HloInstruction* b = d->mutable_operand(1);
  if (!IsRowBroadcast(b, x->shape())) return nullptr;
  m->intermediates.push_back(d);
  m->intermediates.push_back(b);
  HloInstruction* max =
      StripConverts(b->mutable_operand(0), &m->intermediates);
  if (!IsMinorReduce(max, HloOpcode::kMaximum,
                     LiteralUtil::MinValue(F32))) {
    return nullptr;
  }
  m->intermediates.push_back(max);
  if (max->operand(0) != x) return nullptr;
  return x;
}

// Matches S = reduce_add(cvt(exp(D))) behind cvt(.); returns the exp.
HloInstruction* MatchSumOfExp(HloInstruction* s, Match* m) {
  s = StripConverts(s, &m->intermediates);
  if (!IsMinorReduce(s, HloOpcode::kAdd, LiteralUtil::Zero(F32))) {
    return nullptr;
  }
  m->intermediates.push_back(s);
  HloInstruction* e = StripConverts(s->mutable_operand(0), &m->intermediates);
  if (e->opcode() != HloOpcode::kExp) return nullptr;
  return e;
}

bool MatchSoftmax(HloInstruction* root, Match* m) {
  const Shape& shape = root->shape();
  if (!IsDefaultLayoutArray(shape) || !IsSupportedType(shape.element_type()) ||
      shape.dimensions().empty() ||
      shape.dimensions().back() > MetalSoftmaxRewriter::kMaxRowLength ||
      ShapeUtil::ElementsIn(shape) == 0) {
    return false;
  }
  if (root->opcode() == HloOpcode::kDivide) {
    HloInstruction* e = root->mutable_operand(0);
    HloInstruction* b = root->mutable_operand(1);
    if (e->opcode() != HloOpcode::kExp || !IsRowBroadcast(b, shape)) {
      return false;
    }
    m->intermediates.push_back(b);
    if (MatchSumOfExp(b->mutable_operand(0), m) != e) return false;
    m->intermediates.push_back(e);
    m->x = MatchShifted(e->mutable_operand(0), m);
    m->log = false;
  } else if (root->opcode() == HloOpcode::kSubtract) {
    HloInstruction* d = root->mutable_operand(0);
    HloInstruction* b = root->mutable_operand(1);
    if (!IsRowBroadcast(b, shape)) return false;
    m->intermediates.push_back(b);
    HloInstruction* lg = StripConverts(b->mutable_operand(0), &m->intermediates);
    if (lg->opcode() != HloOpcode::kLog) return false;
    m->intermediates.push_back(lg);
    HloInstruction* e = MatchSumOfExp(lg->mutable_operand(0), m);
    if (e == nullptr || e->operand(0) != d) return false;
    m->intermediates.push_back(e);
    m->x = MatchShifted(d, m);
    m->log = true;
  } else {
    return false;
  }
  if (m->x == nullptr || m->x->shape().element_type() != shape.element_type() ||
      !IsDefaultLayoutArray(m->x->shape())) {
    return false;
  }
  // Every intermediate must be used only inside the pattern.
  absl::flat_hash_set<const HloInstruction*> inside(m->intermediates.begin(),
                                                    m->intermediates.end());
  inside.insert(root);
  for (const HloInstruction* h : m->intermediates) {
    if (h == m->x) return false;
    for (const HloInstruction* user : h->users()) {
      if (!inside.contains(user)) return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> MetalSoftmaxRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // Match everything first: replacing an instruction deletes the pattern's
    // now-dead intermediates.
    std::vector<std::pair<HloInstruction*, Match>> matches;
    absl::flat_hash_set<const HloInstruction*> consumed;
    for (HloInstruction* root : comp->MakeInstructionPostOrder()) {
      if (root->opcode() != HloOpcode::kDivide &&
          root->opcode() != HloOpcode::kSubtract) {
        continue;
      }
      Match m;
      if (!MatchSoftmax(root, &m) || consumed.contains(root)) continue;
      if (absl::c_any_of(m.intermediates, [&](const HloInstruction* h) {
            return consumed.contains(h);
          })) {
        continue;
      }
      consumed.insert(m.intermediates.begin(), m.intermediates.end());
      consumed.insert(root);
      matches.push_back({root, std::move(m)});
    }
    for (auto& [root, m] : matches) {
      const int64_t n = root->shape().dimensions().back();
      Shape operand_shape = m.x->shape();
      HloInstruction* call = comp->AddInstruction(
          HloInstruction::CreateCustomCall(
              root->shape(), {m.x}, "metal$softmax", {operand_shape},
              /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
      GpuBackendConfig config;
      config.mutable_custom_call_backend_config()->set_attributes(
          absl::StrCat("{log = ", m.log ? "true" : "false",
                       ", row_length = ", n, " : i64}"));
      TF_RETURN_IF_ERROR(call->set_backend_config(config));
      call->set_metadata(root->metadata());
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(root, call));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
