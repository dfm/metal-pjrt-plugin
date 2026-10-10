// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/rownorm_fusion.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Limits that keep a fusion's kernel small (arguments) and its last loop
// cheap (one store per output).
constexpr int kMaxOutputs = 8;
constexpr int kMaxParameters = 16;
// Cheap producers duplicated into one fusion (constants, iotas, masks).
constexpr int kMaxDuplicated = 32;

bool HasDefaultLayout(const Shape& shape) {
  return shape.IsArray() && shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

// The emitter combines partial results with simdgroup shuffles, which take
// scalars of at most 32 bits (codegen/msl_emitter.cc).
bool IsShuffleType(PrimitiveType type) {
  switch (type) {
    case F32:
    case F16:
    case BF16:
    case S32:
    case U32:
      return true;
    default:
      return false;
  }
}

// Element types of values in a fusion: what the emitter is tested with.
bool IsMemberType(PrimitiveType type) {
  return IsShuffleType(type) || type == PRED;
}

// (a, b) -> a op b for a commutative op the emitter can shuffle-reduce.
bool IsSimpleReducer(const HloComputation* reducer) {
  if (reducer->num_parameters() != 2 || reducer->instruction_count() != 3) {
    return false;
  }
  const HloInstruction* root = reducer->root_instruction();
  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
      break;
    default:
      return false;
  }
  return ShapeUtil::IsScalar(root->shape()) &&
         root->operand(0)->opcode() == HloOpcode::kParameter &&
         root->operand(1)->opcode() == HloOpcode::kParameter &&
         root->operand(0) != root->operand(1);
}

// The value of a reduction's init: a constant, or a convert of one (bf16
// reductions upcast to f32 keep convert(bf16 -inf)).
std::optional<Literal> InitValue(const HloInstruction* init) {
  if (!ShapeUtil::IsScalar(init->shape())) return std::nullopt;
  if (init->opcode() == HloOpcode::kConstant) return init->literal().Clone();
  if (init->opcode() == HloOpcode::kConvert &&
      init->operand(0)->opcode() == HloOpcode::kConstant) {
    absl::StatusOr<Literal> converted =
        init->operand(0)->literal().Convert(init->shape().element_type());
    if (converted.ok()) return *std::move(converted);
  }
  return std::nullopt;
}

// Whether `value` is the identity of the reducer's op: each of the row's
// threads starts from it, so it is combined once per thread, not once.
bool IsIdentity(const Literal& value, HloOpcode op) {
  switch (op) {
    case HloOpcode::kAdd:
      return value.IsAll(0);
    case HloOpcode::kMultiply:
      return value.IsAll(1);
    case HloOpcode::kMaximum:
      return value == LiteralUtil::MinValue(value.shape().element_type());
    case HloOpcode::kMinimum:
      return value == LiteralUtil::MaxValue(value.shape().element_type());
    default:
      return false;
  }
}

// The [rows..., n] shape a fusion works on; [rows...] holds row values.
struct RowSpace {
  std::vector<int64_t> full;

  absl::Span<const int64_t> rows() const {
    return absl::MakeConstSpan(full).first(full.size() - 1);
  }
  int64_t n() const { return full.back(); }
  bool IsFull(const Shape& shape) const {
    return HasDefaultLayout(shape) && absl::c_equal(shape.dimensions(), full);
  }
  bool IsRow(const Shape& shape) const {
    return HasDefaultLayout(shape) && absl::c_equal(shape.dimensions(), rows());
  }
};

// The row space of a reduce over the minor dimension that the emitter takes.
std::optional<RowSpace> RowReduceSpace(const HloInstruction* instr,
                                       int64_t max_row_length) {
  if (instr->opcode() != HloOpcode::kReduce || instr->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* input = instr->operand(0);
  const HloInstruction* init = instr->operand(1);
  const int64_t rank = input->shape().dimensions().size();
  if (rank < 2 || !HasDefaultLayout(input->shape()) ||
      !HasDefaultLayout(instr->shape()) || instr->dimensions().size() != 1 ||
      instr->dimensions(0) != rank - 1 ||
      !IsShuffleType(instr->shape().element_type()) ||
      input->shape().element_type() != instr->shape().element_type() ||
      !IsSimpleReducer(instr->to_apply())) {
    return std::nullopt;
  }
  std::optional<Literal> init_value = InitValue(init);
  if (!init_value.has_value() ||
      !IsIdentity(*init_value,
                  instr->to_apply()->root_instruction()->opcode())) {
    return std::nullopt;
  }
  RowSpace space{std::vector<int64_t>(input->shape().dimensions().begin(),
                                      input->shape().dimensions().end())};
  if (space.n() < 2 || space.n() > max_row_length) return std::nullopt;
  return space;
}

bool IsRowBroadcastDims(absl::Span<const int64_t> dims, int64_t row_rank) {
  for (int64_t i = 0; i < row_rank; ++i) {
    if (i >= static_cast<int64_t>(dims.size()) || dims[i] != i) return false;
  }
  return static_cast<int64_t>(dims.size()) == row_rank;
}

// Elementwise ops a fusion may hold: ones whose operands all have the
// result's dimensions (no implicit scalar broadcast, as clamp allows), with
// no layout change (copy) or called computation (map, fusion).
bool IsAllowedElementwise(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kCompare:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kXor:
      return true;
    default:
      return false;
  }
}

// Whether `instr` can be computed inside a fusion over `space`: at an index
// (rows..., col) for [rows..., n] values, (rows...) for row values.
bool FitsRowSpace(const HloInstruction* instr, const RowSpace& space,
                  int64_t max_row_length) {
  const Shape& shape = instr->shape();
  // Cloning into a fusion drops control dependencies, and the originals
  // could then not be removed.
  if (!shape.IsArray() || instr->HasSideEffect() ||
      !IsMemberType(shape.element_type()) ||
      !instr->control_predecessors().empty() ||
      !instr->control_successors().empty()) {
    return false;
  }
  const bool full = space.IsFull(shape);
  const bool row = space.IsRow(shape);
  switch (instr->opcode()) {
    case HloOpcode::kReduce: {
      std::optional<RowSpace> reduce_space =
          RowReduceSpace(instr, max_row_length);
      return reduce_space.has_value() && reduce_space->full == space.full;
    }
    case HloOpcode::kBroadcast: {
      const Shape& in = instr->operand(0)->shape();
      if (ShapeUtil::IsScalar(in)) return full || row;
      if (!full || !HasDefaultLayout(in)) return false;
      const int64_t rank = space.full.size();
      if (space.IsRow(in) &&
          IsRowBroadcastDims(instr->dimensions(), rank - 1)) {
        return true;
      }
      // A vector along the row, e.g. layer-norm weights. (For [n, n], a
      // [n] operand may be either, told apart by the dimensions.)
      return in.dimensions().size() == 1 && in.dimensions(0) == space.n() &&
             instr->dimensions().size() == 1 &&
             instr->dimensions(0) == rank - 1;
    }
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(shape);
    case HloOpcode::kIota:
      return full || row;
    default:
      if (!IsAllowedElementwise(instr->opcode()) || !(full || row)) {
        return false;
      }
      return absl::c_all_of(instr->operands(), [&](const HloInstruction* op) {
        return IsMemberType(op->shape().element_type()) &&
               (full ? space.IsFull(op->shape()) : space.IsRow(op->shape()));
      });
  }
}

// Constants, iotas, broadcasts of scalars or vectors, and elementwise ops of
// those: recomputed in each fusion that reads them rather than read.
bool IsCheap(const HloInstruction* instr, const RowSpace& space,
             int64_t max_row_length, int depth = 0) {
  if (depth > 8 || !FitsRowSpace(instr, space, max_row_length)) return false;
  switch (instr->opcode()) {
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
      return true;
    case HloOpcode::kBroadcast:
      // Reads at most a row's worth of values per row.
      return true;
    default:
      if (!IsAllowedElementwise(instr->opcode())) return false;
      return absl::c_all_of(instr->operands(), [&](const HloInstruction* op) {
        return IsCheap(op, space, max_row_length, depth + 1);
      });
  }
}

class RegionBuilder {
 public:
  RegionBuilder(RowSpace space, int64_t max_row_length,
                const HloReachabilityMap& reachability)
      : space_(std::move(space)),
        max_row_length_(max_row_length),
        reachability_(reachability) {}

  // Grows a region from `seed`; returns false if it can't be fused.
  bool Build(HloInstruction* seed) {
    Add(seed, /*duplicated=*/false);
    bool changed = true;
    while (changed) {
      changed = false;
      // Consumers, and sibling reductions of the region's inputs.
      std::vector<HloInstruction*> snapshot(order_.begin(), order_.end());
      for (HloInstruction* member : snapshot) {
        if (duplicated_.contains(member)) continue;
        for (HloInstruction* user : member->users()) {
          if (members_.contains(user) ||
              !FitsRowSpace(user, space_, max_row_length_) ||
              ReachableFromRegion(user)) {
            continue;
          }
          changed |= Add(user, /*duplicated=*/false);
        }
      }
      for (HloInstruction* input : ExternalOperands()) {
        if (!space_.IsFull(input->shape())) continue;
        for (HloInstruction* user : input->users()) {
          if (members_.contains(user) ||
              user->opcode() != HloOpcode::kReduce ||
              !FitsRowSpace(user, space_, max_row_length_) ||
              ReachableFromRegion(user)) {
            continue;
          }
          changed |= Add(user, /*duplicated=*/false);
        }
      }
      // Producers used only here, and cheap ones.
      for (HloInstruction* input : ExternalOperands()) {
        if (!FitsRowSpace(input, space_, max_row_length_)) continue;
        // Users that are duplicated members don't count: their originals
        // stay outside and still read `input`.
        if (input->opcode() != HloOpcode::kParameter && !input->IsRoot() &&
            absl::c_all_of(input->users(), [&](HloInstruction* user) {
              return members_.contains(user) && !duplicated_.contains(user);
            })) {
          changed |= Add(input, /*duplicated=*/false);
        } else if (duplicated_.size() < kMaxDuplicated &&
                   IsCheap(input, space_, max_row_length_) &&
                   !ReachableFromRegion(input)) {
          changed |= Add(input, /*duplicated=*/true);
        }
      }
    }
    return Valid();
  }

  const RowSpace& space() const { return space_; }
  const absl::flat_hash_set<HloInstruction*>& members() const {
    return members_;
  }
  const absl::flat_hash_set<HloInstruction*>& duplicated() const {
    return duplicated_;
  }

  // Values used outside the region, in the order they were added.
  // Members used outside the region (or the computation's root).
  std::vector<HloInstruction*> UsedOutside() const {
    std::vector<HloInstruction*> used;
    for (HloInstruction* member : order_) {
      if (duplicated_.contains(member)) continue;
      if (member->IsRoot() ||
          absl::c_any_of(member->users(), [&](const HloInstruction* user) {
            return !members_.contains(user);
          })) {
        used.push_back(member);
      }
    }
    return used;
  }

  // A broadcast of a member used outside: the fusion outputs its operand
  // and the broadcast is redone outside (where it fuses into its users),
  // rather than writing [rows..., n] copies of row values.
  bool IsRebroadcast(const HloInstruction* instr) const {
    return instr->opcode() == HloOpcode::kBroadcast && !instr->IsRoot() &&
           members_.contains(instr->operand(0));
  }

  // The fusion's outputs: the members used outside, with rebroadcasts
  // replaced by their operands.
  std::vector<HloInstruction*> Outputs() const {
    std::vector<HloInstruction*> outputs;
    absl::flat_hash_set<HloInstruction*> seen;
    for (HloInstruction* member : UsedOutside()) {
      HloInstruction* output =
          IsRebroadcast(member) ? member->mutable_operand(0) : member;
      if (seen.insert(output).second) outputs.push_back(output);
    }
    return outputs;
  }

  std::vector<HloInstruction*> ExternalOperands() const {
    std::vector<HloInstruction*> inputs;
    absl::flat_hash_set<HloInstruction*> seen;
    for (HloInstruction* member : order_) {
      for (HloInstruction* operand : member->operands()) {
        if (!members_.contains(operand) && seen.insert(operand).second) {
          inputs.push_back(operand);
        }
      }
    }
    return inputs;
  }

 private:
  bool Add(HloInstruction* instr, bool duplicated) {
    if (!members_.insert(instr).second) return false;
    order_.push_back(instr);
    if (duplicated) duplicated_.insert(instr);
    return true;
  }

  // Whether an operand of `instr` outside the region depends on the region:
  // fusing `instr` would then create a cycle. Duplicated members don't
  // count: the originals stay outside.
  bool ReachableFromRegion(const HloInstruction* instr) const {
    for (const HloInstruction* operand : instr->operands()) {
      if (members_.contains(operand)) continue;
      if (DependsOnRegion(operand)) return true;
    }
    return false;
  }

  bool DependsOnRegion(const HloInstruction* instr) const {
    for (const HloInstruction* member : order_) {
      if (!duplicated_.contains(member) &&
          reachability_.IsReachable(member, instr)) {
        return true;
      }
    }
    return false;
  }

  bool Valid() const {
    // No path region -> outside -> region.
    for (const HloInstruction* input : ExternalOperands()) {
      if (DependsOnRegion(input)) {
        VLOG(2) << "rownorm: cycle through " << input->name();
        return false;
      }
    }
    std::vector<HloInstruction*> outputs = Outputs();
    if (outputs.empty() || outputs.size() > kMaxOutputs ||
        ExternalOperands().size() > kMaxParameters) {
      return false;
    }
    for (const HloInstruction* output : outputs) {
      if (!space_.IsFull(output->shape()) && !space_.IsRow(output->shape())) {
        return false;
      }
    }
    // A diamond must close: an elementwise op along the row that combines
    // a row reduction's result with row data (not a broadcast). Otherwise
    // XLA's reduction emitter does as well.
    absl::flat_hash_map<const HloInstruction*, bool> memo;
    std::function<bool(const HloInstruction*)> depends_on_reduce =
        [&](const HloInstruction* instr) -> bool {
      if (!members_.contains(instr)) return false;
      if (instr->opcode() == HloOpcode::kReduce) return true;
      auto it = memo.find(instr);
      if (it != memo.end()) return it->second;
      bool result = absl::c_any_of(instr->operands(), depends_on_reduce);
      memo[instr] = result;
      return result;
    };
    auto is_row_data = [&](const HloInstruction* instr) {
      return space_.IsFull(instr->shape()) &&
             instr->opcode() != HloOpcode::kBroadcast &&
             instr->opcode() != HloOpcode::kIota;
    };
    return absl::c_any_of(order_, [&](const HloInstruction* member) {
      if (!space_.IsFull(member->shape()) ||
          !IsAllowedElementwise(member->opcode())) {
        return false;
      }
      return absl::c_any_of(member->operands(), depends_on_reduce) &&
             absl::c_any_of(member->operands(), is_row_data);
    });
  }

  RowSpace space_;
  int64_t max_row_length_;
  const HloReachabilityMap& reachability_;
  absl::flat_hash_set<HloInstruction*> members_;
  absl::flat_hash_set<HloInstruction*> duplicated_;
  std::vector<HloInstruction*> order_;
};

// Replaces the region with a kCustom fusion; returns the fusion.
absl::StatusOr<HloInstruction*> FuseRegion(HloComputation* computation,
                                           const RegionBuilder& region) {
  const auto& members = region.members();
  std::vector<HloInstruction*> post_order;
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (members.contains(instr)) post_order.push_back(instr);
  }
  std::vector<HloInstruction*> outputs = region.Outputs();

  HloComputation::Builder builder("metal_rownorm_computation");
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> fused;
  std::vector<HloInstruction*> fusion_operands;
  auto parameter_for = [&](HloInstruction* operand) {
    auto it = fused.find(operand);
    if (it != fused.end()) return it->second;
    const int64_t index = fusion_operands.size();
    HloInstruction* parameter =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, operand->shape(), absl::StrCat("p", index)));
    fusion_operands.push_back(operand);
    fused[operand] = parameter;
    return parameter;
  };
  for (HloInstruction* instr : post_order) {
    std::vector<HloInstruction*> operands;
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(members.contains(operand) ? fused.at(operand)
                                                   : parameter_for(operand));
    }
    fused[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), operands));
  }
  HloInstruction* root;
  if (outputs.size() == 1) {
    root = fused.at(outputs[0]);
  } else {
    std::vector<HloInstruction*> elements;
    for (HloInstruction* output : outputs) elements.push_back(fused.at(output));
    root = builder.AddInstruction(HloInstruction::CreateTuple(elements));
  }
  HloComputation* fused_computation =
      computation->parent()->AddComputationAndUnifyNamesAndIds(
          builder.Build(root), /*is_entry=*/false);
  HloInstruction* fusion = computation->AddInstruction(
      HloInstruction::CreateFusion(root->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   fusion_operands, fused_computation));
  computation->parent()->SetAndUniquifyInstrName(fusion, "metal_rownorm");
  // For profiles: the op_name of the first reduction (e.g. jax's softmax).
  for (const HloInstruction* instr : post_order) {
    if (instr->opcode() == HloOpcode::kReduce) {
      fusion->set_metadata(instr->metadata());
      break;
    }
  }
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kCustomFusionKind));
  backend_config.mutable_custom_fusion_config()->set_name(
      std::string(kMetalRowNormFusionName));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> value_of;
  for (int64_t i = 0; i < static_cast<int64_t>(outputs.size()); ++i) {
    HloInstruction* value =
        outputs.size() == 1
            ? fusion
            : computation->AddInstruction(
                  HloInstruction::CreateGetTupleElement(fusion, i));
    value->set_metadata(outputs[i]->metadata());
    value_of[outputs[i]] = value;
  }
  for (HloInstruction* member : region.UsedOutside()) {
    HloInstruction* value;
    if (region.IsRebroadcast(member)) {
      value = computation->AddInstruction(member->CloneWithNewOperands(
          member->shape(), {value_of.at(member->operand(0))}));
    } else {
      value = value_of.at(member);
    }
    std::vector<HloInstruction*> outside_users;
    for (HloInstruction* user : member->users()) {
      if (!members.contains(user)) outside_users.push_back(user);
    }
    TF_RETURN_IF_ERROR(member->ReplaceUsesWith(outside_users, value));
    if (member->IsRoot()) computation->set_root_instruction(value);
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    if ((*it)->user_count() == 0 && !(*it)->IsRoot()) {
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(*it));
    }
  }
  return fusion;
}

}  // namespace

bool IsMetalRowNormFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion ||
      instr.fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return false;
  }
  std::optional<std::string> name = GetCustomFusionConfigName(&instr);
  return name.has_value() && *name == kMetalRowNormFusionName;
}

absl::StatusOr<bool> MetalRowNormFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // unique_ids of reductions that seeded no fusion.
    absl::flat_hash_set<int> tried;
    while (true) {
      // The graph changes with every fusion, so the reachability map is
      // rebuilt for each.
      std::unique_ptr<HloReachabilityMap> reachability =
          HloReachabilityMap::Build(computation);
      std::optional<RegionBuilder> found;
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (tried.contains(instr->unique_id())) continue;
        std::optional<RowSpace> space = RowReduceSpace(instr, max_row_length_);
        if (!space.has_value() || space->n() < min_row_length_) continue;
        tried.insert(instr->unique_id());
        RegionBuilder region(*space, max_row_length_, *reachability);
        if (region.Build(instr)) {
          found.emplace(std::move(region));
          break;
        }
      }
      if (!found.has_value()) break;
      TF_ASSIGN_OR_RETURN(HloInstruction * fusion,
                          FuseRegion(computation, *found));
      VLOG(2) << "rownorm: " << fusion->ToString();
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
