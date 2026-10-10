// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/sort_expander.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Builds elementwise HLO over rank-1 arrays of length `m` inside `comp`.
class FlatBuilder {
 public:
  FlatBuilder(HloComputation* comp, int64_t m, PrimitiveType index_type)
      : comp_(comp), m_(m), index_type_(index_type) {}

  Shape ArrayShape(PrimitiveType t) const {
    return ShapeUtil::MakeShape(t, {m_});
  }

  HloInstruction* Add(std::unique_ptr<HloInstruction> instr) {
    return comp_->AddInstruction(std::move(instr));
  }

  HloInstruction* ScalarIndex(int64_t v) {
    Literal lit = LiteralUtil::CreateR0<int64_t>(v);
    if (index_type_ != S64) lit = lit.Convert(index_type_).value();
    return Add(HloInstruction::CreateConstant(std::move(lit)));
  }

  HloInstruction* Broadcast(HloInstruction* scalar) {
    return Add(HloInstruction::CreateBroadcast(
        ArrayShape(scalar->shape().element_type()), scalar, {}));
  }

  HloInstruction* Binary(HloOpcode op, HloInstruction* a, HloInstruction* b) {
    return Add(HloInstruction::CreateBinary(a->shape(), op, a, b));
  }

  HloInstruction* Compare(Comparison::Direction dir, HloInstruction* a,
                          HloInstruction* b) {
    Shape shape = a->shape();
    shape.set_element_type(PRED);
    return Add(HloInstruction::CreateCompare(shape, a, b, dir));
  }

  HloInstruction* Not(HloInstruction* a) {
    return Add(HloInstruction::CreateUnary(a->shape(), HloOpcode::kNot, a));
  }

  HloInstruction* Select(HloInstruction* p, HloInstruction* t,
                         HloInstruction* f) {
    return Add(HloInstruction::CreateTernary(t->shape(), HloOpcode::kSelect, p,
                                             t, f));
  }

  HloComputation* comp() const { return comp_; }
  int64_t m() const { return m_; }
  PrimitiveType index_type() const { return index_type_; }

 private:
  HloComputation* comp_;
  int64_t m_;
  PrimitiveType index_type_;
};

// Returns true if every instruction of `comparator` can be applied
// elementwise to arrays by cloning with an array shape.
bool ComparatorIsElementwise(const HloComputation* comparator) {
  for (const HloInstruction* instr : comparator->instructions()) {
    if (!ShapeUtil::IsScalar(instr->shape())) return false;
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
        continue;
      default:
        break;
    }
    if (!instr->IsElementwise() || instr->HasSideEffect()) return false;
    if (instr->opcode() == HloOpcode::kRng) return false;
  }
  return true;
}

// Applies `comparator` elementwise: comparator(lhs[i], rhs[i]) for each
// operand i, with the sort convention that parameter 2*i is lhs of operand i
// and 2*i+1 is rhs of operand i. Returns a PRED[m] array.
absl::StatusOr<HloInstruction*> ApplyComparator(
    FlatBuilder& b, HloComputation* comparator,
    absl::Span<HloInstruction* const> lhs,
    absl::Span<HloInstruction* const> rhs) {
  std::vector<HloInstruction*> args;
  args.reserve(2 * lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    args.push_back(lhs[i]);
    args.push_back(rhs[i]);
  }
  if (!ComparatorIsElementwise(comparator)) {
    return MakeMapHlo(args, comparator);
  }
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> mapped;
  for (HloInstruction* instr : comparator->MakeInstructionPostOrder()) {
    HloInstruction* out = nullptr;
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
        out = args.at(instr->parameter_number());
        break;
      case HloOpcode::kConstant:
        out = b.Broadcast(b.Add(instr->Clone()));
        break;
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
        out = mapped.at(instr->operand(0));
        if (out->shape().element_type() != instr->shape().element_type()) {
          return absl::InternalError(absl::StrCat(
              "unexpected type-changing op in sort comparator: ",
              instr->ToString(), metal_pjrt::kReportBug));
        }
        break;
      default: {
        std::vector<HloInstruction*> operands;
        for (const HloInstruction* op : instr->operands()) {
          operands.push_back(mapped.at(op));
        }
        out = b.Add(instr->CloneWithNewOperands(
            b.ArrayShape(instr->shape().element_type()), operands));
        break;
      }
    }
    mapped[instr] = out;
  }
  HloInstruction* root = mapped.at(comparator->root_instruction());
  if (root->shape().element_type() != PRED) {
    return absl::InvalidArgumentError("sort comparator must return PRED");
  }
  return root;
}

// Strict total order "a < b" built from the user comparator: ties (both or
// neither compare less, so a non-strict comparator such as LE works too) are
// broken by the original flat index, and padded elements (original row
// position >= n) sort after all real elements.
absl::StatusOr<HloInstruction*> TotalLess(
    FlatBuilder& b, HloComputation* comparator, int64_t n, int64_t pow2,
    absl::Span<HloInstruction* const> a_vals, HloInstruction* a_idx,
    absl::Span<HloInstruction* const> b_vals, HloInstruction* b_idx) {
  TF_ASSIGN_OR_RETURN(HloInstruction * c_ab,
                      ApplyComparator(b, comparator, a_vals, b_vals));
  TF_ASSIGN_OR_RETURN(HloInstruction * c_ba,
                      ApplyComparator(b, comparator, b_vals, a_vals));
  HloInstruction* idx_lt =
      b.Compare(Comparison::Direction::kLt, a_idx, b_idx);
  HloInstruction* same = b.Compare(Comparison::Direction::kEq, c_ab, c_ba);
  HloInstruction* tie = b.Binary(HloOpcode::kAnd, same, idx_lt);
  HloInstruction* valid_less = b.Binary(
      HloOpcode::kOr, b.Binary(HloOpcode::kAnd, c_ab, b.Not(c_ba)), tie);
  if (n == pow2) return valid_less;

  HloInstruction* row_mask = b.Broadcast(b.ScalarIndex(pow2 - 1));
  HloInstruction* n_b = b.Broadcast(b.ScalarIndex(n));
  HloInstruction* va = b.Compare(Comparison::Direction::kLt,
                                 b.Binary(HloOpcode::kAnd, a_idx, row_mask),
                                 n_b);
  HloInstruction* vb = b.Compare(Comparison::Direction::kLt,
                                 b.Binary(HloOpcode::kAnd, b_idx, row_mask),
                                 n_b);
  HloInstruction* both = b.Binary(HloOpcode::kAnd, va, vb);
  HloInstruction* differ = b.Compare(Comparison::Direction::kNe, va, vb);
  return b.Select(both, valid_less, b.Select(differ, va, idx_lt));
}

struct NetworkState {
  HloInstruction* k;  // current bitonic block size (scalar)
  HloInstruction* j;  // current substage distance (scalar)
  std::vector<HloInstruction*> arrays;  // operands..., original index
};

// Emits one compare-and-swap substage of the bitonic network into b.comp()
// and advances (k, j) to the next substage.
absl::StatusOr<NetworkState> EmitStep(FlatBuilder& b,
                                      HloComputation* comparator,
                                      int64_t num_operands, int64_t n,
                                      int64_t pow2, const NetworkState& in) {
  const PrimitiveType itype = b.index_type();
  const int64_t m = b.m();
  const std::vector<HloInstruction*>& self = in.arrays;

  HloInstruction* pos =
      b.Add(HloInstruction::CreateIota(b.ArrayShape(itype), 0));
  HloInstruction* jb = b.Broadcast(in.j);
  HloInstruction* km =
      b.Broadcast(b.Binary(HloOpcode::kAnd, in.k, b.ScalarIndex(pow2 - 1)));
  HloInstruction* partner_pos = b.Binary(HloOpcode::kXor, pos, jb);
  HloInstruction* partner_idx = b.Add(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(itype, {m, 1}), partner_pos));

  GatherDimensionNumbers dnums = HloGatherInstruction::MakeGatherDimNumbers(
      /*offset_dims=*/{1}, /*collapsed_slice_dims=*/{},
      /*start_index_map=*/{0}, /*index_vector_dim=*/1);
  std::vector<HloInstruction*> partner;
  for (HloInstruction* s : self) {
    PrimitiveType et = s->shape().element_type();
    HloInstruction* g = b.Add(HloInstruction::CreateGather(
        ShapeUtil::MakeShape(et, {m, 1}), s, partner_idx, dnums,
        /*slice_sizes=*/{1}, /*indices_are_sorted=*/false));
    partner.push_back(
        b.Add(HloInstruction::CreateReshape(b.ArrayShape(et), g)));
  }

  HloInstruction* zero = b.Broadcast(b.ScalarIndex(0));
  HloInstruction* lower = b.Compare(Comparison::Direction::kEq,
                                    b.Binary(HloOpcode::kAnd, pos, jb), zero);
  HloInstruction* asc = b.Compare(Comparison::Direction::kEq,
                                  b.Binary(HloOpcode::kAnd, pos, km), zero);
  HloInstruction* want_min = b.Compare(Comparison::Direction::kEq, lower, asc);

  absl::Span<HloInstruction* const> self_span(self);
  absl::Span<HloInstruction* const> partner_span(partner);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * self_less,
      TotalLess(b, comparator, n, pow2, self_span.first(num_operands),
                self.back(), partner_span.first(num_operands),
                partner.back()));
  // Keep self iff (want_min && self<partner) || (!want_min && partner<self);
  // with a strict total order and distinct indices this is want_min == less.
  HloInstruction* take_partner =
      b.Compare(Comparison::Direction::kNe, want_min, self_less);

  NetworkState out;
  HloInstruction* one = b.ScalarIndex(1);
  HloInstruction* j_is_one = b.Compare(Comparison::Direction::kEq, in.j, one);
  out.k = b.Select(j_is_one, b.Binary(HloOpcode::kShiftLeft, in.k, one), in.k);
  out.j = b.Select(j_is_one, in.k,
                   b.Binary(HloOpcode::kShiftRightLogical, in.j, one));
  for (size_t i = 0; i < self.size(); ++i) {
    out.arrays.push_back(b.Select(take_partner, partner[i], self[i]));
  }
  return out;
}

// Builds the while-loop body, which runs two substages per iteration: the
// gathers make each step's output unable to share its input's buffer, so a
// single-step body would need a copy per array per iteration. With two steps
// the second one can write into the (dead) loop-carried buffers.
// State layout: (iteration, k, j, operand_0, ..., operand_{K-1}, orig_index).
absl::StatusOr<HloComputation*> BuildBody(HloModule* module,
                                          const Shape& state_shape,
                                          HloComputation* comparator,
                                          int64_t num_operands, int64_t n,
                                          int64_t pow2, int64_t m,
                                          PrimitiveType itype) {
  HloComputation::Builder builder("metal_sort_body");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, state_shape, "state"));
  HloComputation* body = module->AddEmbeddedComputation(builder.Build());
  HloInstruction* state = body->parameter_instruction(0);
  FlatBuilder b(body, m, itype);

  auto gte = [&](int64_t i) {
    return b.Add(HloInstruction::CreateGetTupleElement(state, i));
  };
  HloInstruction* iter = gte(0);
  NetworkState s;
  s.k = gte(1);
  s.j = gte(2);
  for (int64_t i = 0; i < num_operands + 1; ++i) s.arrays.push_back(gte(3 + i));
  for (int step = 0; step < 2; ++step) {
    TF_ASSIGN_OR_RETURN(s, EmitStep(b, comparator, num_operands, n, pow2, s));
  }
  std::vector<HloInstruction*> outs = {
      b.Binary(HloOpcode::kAdd, iter, b.ScalarIndex(1)), s.k, s.j};
  outs.insert(outs.end(), s.arrays.begin(), s.arrays.end());
  body->set_root_instruction(b.Add(HloInstruction::CreateTuple(outs)));
  return body;
}

HloComputation* BuildCondition(HloModule* module, const Shape& state_shape,
                               int64_t trip_count, PrimitiveType itype) {
  HloComputation::Builder builder("metal_sort_cond");
  HloInstruction* state = builder.AddInstruction(
      HloInstruction::CreateParameter(0, state_shape, "state"));
  HloInstruction* t = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(state, 0));
  Literal lit = LiteralUtil::CreateR0<int64_t>(trip_count);
  if (itype != S64) lit = lit.Convert(itype).value();
  HloInstruction* limit =
      builder.AddInstruction(HloInstruction::CreateConstant(std::move(lit)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), t, limit, Comparison::Direction::kLt));
  return module->AddEmbeddedComputation(builder.Build());
}

absl::StatusOr<HloInstruction*> ExpandSort(HloSortInstruction* sort) {
  HloComputation* comp = sort->parent();
  HloModule* module = comp->parent();
  const int64_t num_operands = sort->operand_count();
  const Shape& shape = sort->operand(0)->shape();
  const int64_t rank = shape.dimensions().size();
  const int64_t dim = sort->sort_dimension();
  const int64_t n = shape.dimensions(dim);
  const int64_t total = ShapeUtil::ElementsIn(shape);

  auto make_result = [&](std::vector<HloInstruction*> outs) -> HloInstruction* {
    if (sort->shape().IsTuple()) {
      return comp->AddInstruction(HloInstruction::CreateTuple(outs));
    }
    return outs[0];
  };

  if (n <= 1 || total == 0) {
    std::vector<HloInstruction*> outs(sort->operands().begin(),
                                      sort->operands().end());
    return make_result(std::move(outs));
  }

  int64_t pow2 = 1;
  int log2 = 0;
  while (pow2 < n) {
    pow2 <<= 1;
    ++log2;
  }
  const int64_t batch = total / n;
  const int64_t m = batch * pow2;
  const PrimitiveType itype =
      m < std::numeric_limits<int32_t>::max() ? S32 : S64;
  const int64_t num_steps = static_cast<int64_t>(log2) * (log2 + 1) / 2;

  // Move the sort dimension to the minor-most position.
  std::vector<int64_t> perm;
  for (int64_t d = 0; d < rank; ++d) {
    if (d != dim) perm.push_back(d);
  }
  perm.push_back(dim);
  std::vector<int64_t> inv_perm(rank);
  for (int64_t d = 0; d < rank; ++d) inv_perm[perm[d]] = d;
  const bool needs_transpose = dim != rank - 1;

  PaddingConfig pad_config;
  pad_config.add_dimensions();
  auto* pad_dim = pad_config.add_dimensions();
  pad_dim->set_edge_padding_high(pow2 - n);

  FlatBuilder outer(comp, m, itype);
  NetworkState s;
  s.k = outer.ScalarIndex(2);
  s.j = outer.ScalarIndex(1);
  std::vector<int64_t> transposed_dims;
  for (int64_t d : perm) transposed_dims.push_back(shape.dimensions(d));
  for (HloInstruction* op : sort->operands()) {
    HloInstruction* x = op;
    if (needs_transpose) {
      TF_ASSIGN_OR_RETURN(x, MakeTransposeHlo(x, perm));
    }
    TF_ASSIGN_OR_RETURN(x, MakeReshapeHlo({batch, n}, x));
    if (pow2 != n) {
      HloInstruction* zero = comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(op->shape().element_type())));
      TF_ASSIGN_OR_RETURN(x, MakePadHlo(x, zero, pad_config));
    }
    TF_ASSIGN_OR_RETURN(x, MakeReshapeHlo({m}, x));
    s.arrays.push_back(x);
  }
  s.arrays.push_back(comp->AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(itype, {m}), 0)));

  // Small sort dimensions (up to kMaxUnrolledSortDim = 64 elements, 21
  // substages) are emitted
  // straight-line: each substage becomes a fusion, and the whole sort then
  // runs inside one command buffer with no while loop (a while loop costs a
  // thunk-level loop per sort, which dominated batched argsorts of a few
  // elements per row, e.g. LU pivot inversion in jnp.linalg.solve).
  // Larger sorts run the two-substages-per-iteration while loop below, with an
  // odd substage count leaving one step to run before it.
  constexpr int64_t kMaxUnrolledSteps = 21;  // log2(64) * (log2(64) + 1) / 2
  static_assert(kMaxUnrolledSortDim == 64);
  const int64_t pre_steps =
      num_steps <= kMaxUnrolledSteps ? num_steps : num_steps % 2;
  // Keep each substage its own fusion, reading materialized arrays.
  // Otherwise XLA fuses the whole network into every consumer (a gather by
  // the argsort, say), which then recomputes each earlier substage per
  // element, twice per level (select(swap, gather(prev, f ^ j), prev)):
  // 2^steps calls, about 2M per element at 64. The Metal compiler also
  // miscompiles a substage that inlines a computed input twice (self and
  // partner): that nested form gave wrong columns for
  // v[:, jnp.argsort(s, descending=True)], and the first substage reading
  // its padded, reversed input through two inlined calls gave a wrong
  // stable descending argsort of 8-bit keys. So the inputs are materialized
  // too, not only each substage's outputs. XLA:GPU removes optimization
  // barriers only after scheduling.
  auto materialize = [&](std::vector<HloInstruction*>& arrays) -> absl::Status {
    HloInstruction* tuple =
        comp->AddInstruction(HloInstruction::CreateTuple(arrays));
    HloInstruction* barrier = comp->AddInstruction(HloInstruction::CreateUnary(
        tuple->shape(), HloOpcode::kOptimizationBarrier, tuple));
    for (int64_t i = 0; i < static_cast<int64_t>(arrays.size()); ++i) {
      TF_ASSIGN_OR_RETURN(arrays[i], MakeGetTupleElementHlo(barrier, i));
    }
    return absl::OkStatus();
  };
  if (pre_steps > 0) TF_RETURN_IF_ERROR(materialize(s.arrays));
  for (int64_t step = 0; step < pre_steps; ++step) {
    TF_ASSIGN_OR_RETURN(s, EmitStep(outer, sort->to_apply(), num_operands, n,
                                    pow2, s));
    TF_RETURN_IF_ERROR(materialize(s.arrays));
  }
  std::vector<HloInstruction*> sorted(s.arrays.begin(),
                                      s.arrays.begin() + num_operands);
  if (num_steps > kMaxUnrolledSteps) {
    std::vector<HloInstruction*> init = {outer.ScalarIndex(0), s.k, s.j};
    init.insert(init.end(), s.arrays.begin(), s.arrays.end());
    HloInstruction* init_tuple =
        comp->AddInstruction(HloInstruction::CreateTuple(init));
    const Shape& state_shape = init_tuple->shape();
    TF_ASSIGN_OR_RETURN(
        HloComputation * body,
        BuildBody(module, state_shape, sort->to_apply(), num_operands, n, pow2,
                  m, itype));
    HloComputation* cond =
        BuildCondition(module, state_shape, num_steps / 2, itype);
    HloInstruction* loop = comp->AddInstruction(
        HloInstruction::CreateWhile(state_shape, cond, body, init_tuple));
    for (int64_t i = 0; i < num_operands; ++i) {
      TF_ASSIGN_OR_RETURN(sorted[i], MakeGetTupleElementHlo(loop, 3 + i));
    }
  }

  std::vector<HloInstruction*> outs;
  for (int64_t i = 0; i < num_operands; ++i) {
    TF_ASSIGN_OR_RETURN(HloInstruction * x,
                        MakeReshapeHlo({batch, pow2}, sorted[i]));
    if (pow2 != n) {
      TF_ASSIGN_OR_RETURN(x, MakeSliceHlo(x, {0, 0}, {batch, n}, {1, 1}));
    }
    TF_ASSIGN_OR_RETURN(x, MakeReshapeHlo(transposed_dims, x));
    if (needs_transpose) {
      TF_ASSIGN_OR_RETURN(x, MakeTransposeHlo(x, inv_perm));
    }
    outs.push_back(x);
  }
  return make_result(std::move(outs));
}

}  // namespace

absl::StatusOr<bool> MetalSortExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloSortInstruction*> sorts;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kSort) continue;
      auto* sort = Cast<HloSortInstruction>(instr);
      const Shape& shape = sort->operand(0)->shape();
      if (shape.dimensions(sort->sort_dimension()) > max_sort_dim_ ||
          ShapeUtil::ElementsIn(shape) <= min_elements_) {
        continue;
      }
      sorts.push_back(sort);
    }
  }
  for (HloSortInstruction* sort : sorts) {
    TF_ASSIGN_OR_RETURN(HloInstruction * replacement, ExpandSort(sort));
    TF_RETURN_IF_ERROR(sort->parent()->ReplaceInstruction(sort, replacement));
  }
  return !sorts.empty();
}

}  // namespace gpu
}  // namespace xla
