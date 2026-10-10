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
      init->opcode() != HloOpcode::kConstant ||
      !IsShuffleType(instr->shape().element_type()) ||
      !IsSimpleReducer(instr->to_apply())) {
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

// Whether `instr` can be computed inside a fusion over `space`: at an index
// (rows..., col) for [rows..., n] values, (rows...) for row values.
bool FitsRowSpace(const HloInstruction* instr, const RowSpace& space,
                  int64_t max_row_length) {
  const Shape& shape = instr->shape();
  if (!shape.IsArray() || instr->HasSideEffect()) return false;
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
      if (space.IsRow(in)) return IsRowBroadcastDims(instr->dimensions(), rank - 1);
      // A vector along the row, e.g. layer-norm weights.
      return in.dimensions().size() == 1 && in.dimensions(0) == space.n() &&
             instr->dimensions().size() == 1 &&
             instr->dimensions(0) == rank - 1;
    }
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(shape);
    case HloOpcode::kIota:
      return full || row;
    default:
      return instr->IsElementwise() && (full || row);
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
      if (!instr->IsElementwise()) return false;
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
        if (input->opcode() != HloOpcode::kParameter && !input->IsRoot() &&
            absl::c_all_of(input->users(), [&](const HloInstruction* user) {
              return members_.contains(user);
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
  std::vector<HloInstruction*> Outputs() const {
    std::vector<HloInstruction*> outputs;
    for (HloInstruction* member : order_) {
      if (duplicated_.contains(member)) continue;
      if (member->IsRoot() ||
          absl::c_any_of(member->users(), [&](const HloInstruction* user) {
            return !members_.contains(user);
          })) {
        outputs.push_back(member);
      }
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
    // Something along the row must depend on a row reduction: otherwise
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
    return absl::c_any_of(order_, [&](const HloInstruction* member) {
      return space_.IsFull(member->shape()) && depends_on_reduce(member);
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
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kCustomFusionKind));
  backend_config.mutable_custom_fusion_config()->set_name(
      std::string(kMetalRowNormFusionName));
  TF_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  for (int64_t i = 0; i < static_cast<int64_t>(outputs.size()); ++i) {
    HloInstruction* output = outputs[i];
    HloInstruction* value =
        outputs.size() == 1
            ? fusion
            : computation->AddInstruction(
                  HloInstruction::CreateGetTupleElement(fusion, i));
    std::vector<HloInstruction*> outside_users;
    for (HloInstruction* user : output->users()) {
      if (!members.contains(user)) outside_users.push_back(user);
    }
    TF_RETURN_IF_ERROR(output->ReplaceUsesWith(outside_users, value));
    if (output->IsRoot()) computation->set_root_instruction(value);
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
        if (!space.has_value()) continue;
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
