// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/codegen/rownorm_emitter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "metal_pjrt/compiler/passes/rownorm_fusion.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/emitters/utils.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::Value;
using mlir::ValueRange;
using HloValueMap =
    absl::flat_hash_map<const HloInstruction*, SmallVector<Value>>;

// Elements per thread to aim for: each loop re-reads its row, so more
// threads per row (fewer elements each) hide more latency.
constexpr int64_t kElementsPerThread = 8;
constexpr int64_t kMaxThreadsPerBlock = 1024;
constexpr int64_t kThreadsPerBlockTarget = 256;

// shuffle_reduce leaves the result in lane 0 (it shuffles down); this hands
// lane 0's value to every lane of the simdgroup. 16-bit values travel as
// 32-bit integers.
Value BroadcastLane0(ImplicitLocOpBuilder& b, Value value, int64_t warp_size) {
  mlir::Type type = value.getType();
  const int bits = type.getIntOrFloatBitWidth();
  Value v = value;
  if (bits < 32) {
    v = mlir::arith::BitcastOp::create(b, b.getIntegerType(bits), v);
    v = mlir::arith::ExtUIOp::create(b, b.getI32Type(), v);
  }
  Value lane = mlir::arith::ConstantIntOp::create(b, 0, 32);
  Value width = mlir::arith::ConstantIntOp::create(b, warp_size, 32);
  v = mlir::gpu::ShuffleOp::create(b, v, lane, width,
                                   mlir::gpu::ShuffleMode::IDX)
          .getShuffleResult();
  if (bits < 32) {
    v = mlir::arith::TruncIOp::create(b, b.getIntegerType(bits), v);
    v = mlir::arith::BitcastOp::create(b, type, v);
  }
  return v;
}

}  // namespace

MetalRowNormEmitter::MetalRowNormEmitter(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  const HloComputation* computation =
      analysis.fusion_root(0).instruction().parent();
  for (const HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kReduce) reductions_.push_back(instr);
  }
  CHECK(!reductions_.empty()) << computation->ToString();
  const Shape& input = reductions_.front()->operand(0)->shape();
  full_dims_.assign(input.dimensions().begin(), input.dimensions().end());
  row_dims_.assign(full_dims_.begin(), full_dims_.end() - 1);
  n_ = full_dims_.back();
  rows_ = Product(row_dims_);
  // Values may factor the rows differently ([1024, n] and [4, 256, n]);
  // the loops index the first reduction's input and map to each root's
  // shape (GetBitcastMap).
  for (const HloInstructionAdaptor& root : analysis.fusion_roots()) {
    const HloInstruction* instr = &root.instruction();
    absl::Span<const int64_t> dims = instr->shape().dimensions();
    if (dims.size() >= 2 && dims.back() == n_ &&
        Product(dims) == rows_ * n_) {
      full_roots_.push_back(instr);
    } else {
      CHECK_EQ(Product(dims), rows_) << instr->ToString();
      row_roots_.push_back(instr);
    }
  }

  input_epilogue_[reductions_.front()->operand(0)] = -1;
  for (int k = 1; k < static_cast<int>(reductions_.size()); ++k) {
    const HloInstruction* input = reductions_[k]->operand(0);
    if (input_epilogue_.contains(input)) continue;
    input_epilogue_[input] = input_epilogues_.size();
    input_epilogues_.push_back({k, input});
  }
  for (const HloInstruction* root : full_roots_) {
    if (!input_epilogue_.contains(root)) epilogue_full_roots_.push_back(root);
  }
  for (const HloInstruction* root : row_roots_) {
    if (root->opcode() != HloOpcode::kReduce) {
      epilogue_row_roots_.push_back(root);
    }
  }
  full_epilogue_ = input_epilogues_.size();
  row_epilogue_ = full_epilogue_ + 1;

  warp_size_ = analysis.device_info().threads_per_warp();
  threads_per_row_ = std::clamp(
      RoundUpTo(CeilOfRatio(n_, kElementsPerThread), warp_size_), warp_size_,
      kMaxThreadsPerBlock);
  rows_per_block_ = std::max<int64_t>(
      1, std::min(rows_, kThreadsPerBlockTarget / threads_per_row_));
  num_blocks_ = CeilOfRatio(rows_, rows_per_block_);
  gpu_blocks_ = MaybeSplitGridDimensionX(threads_per_row_ * rows_per_block_,
                                         num_blocks_, analysis.device_info());
}

LaunchDimensions MetalRowNormEmitter::launch_dimensions() const {
  return {se::BlockDim(gpu_blocks_[0], gpu_blocks_[1], 1),
          se::ThreadDim(threads_per_row_ * rows_per_block_, 1, 1)};
}

IndexingMap MetalRowNormEmitter::ThreadMap(
    MLIRContext* ctx, llvm::ArrayRef<SymbolicExpr> results,
    absl::Span<const int64_t> symbol_sizes) const {
  return IndexingMap{
      SymbolicMap::Get(ctx, 6, symbol_sizes.size(), llvm::to_vector(results)),
      DimVarsFromGPUGrid({threads_per_row_ * rows_per_block_, 1, 1,
                          static_cast<int64_t>(gpu_blocks_[0]),
                          static_cast<int64_t>(gpu_blocks_[1]), 1}),
      RangeVarsFromTensorSizes(symbol_sizes),
      /*rt_vars=*/{}};
}

SymbolicExpr MetalRowNormEmitter::RowIndex(MLIRContext* ctx) const {
  SymbolicExpr thread = CreateDimExpr(0, ctx);
  SymbolicExpr block = CreateDimExpr(3, ctx) +
                       CreateDimExpr(4, ctx) * static_cast<int64_t>(gpu_blocks_[0]);
  return block * rows_per_block_ + thread.floorDiv(threads_per_row_);
}

IndexingMap MetalRowNormEmitter::RowLoopMap(MLIRContext* ctx) const {
  SymbolicExpr row = RowIndex(ctx);
  SymbolicExpr lane = CreateDimExpr(0, ctx) % threads_per_row_;
  SymbolicExpr col =
      CreateSymbolExpr(0, /*num_dims=*/6, ctx) * threads_per_row_ + lane;
  SmallVector<SymbolicExpr> results = DelinearizeInBoundsIndex(row, row_dims_);
  results.push_back(col);
  IndexingMap map =
      ThreadMap(ctx, results, {CeilOfRatio(n_, threads_per_row_)});
  map.AddConstraint(row, {0, rows_ - 1});
  map.AddConstraint(col, {0, n_ - 1});
  map.Simplify();
  return map;
}

IndexingMap MetalRowNormEmitter::RowOutputMap(MLIRContext* ctx) const {
  SymbolicExpr row = RowIndex(ctx);
  SmallVector<SymbolicExpr> results = DelinearizeInBoundsIndex(row, row_dims_);
  IndexingMap map = ThreadMap(ctx, results, {});
  map.AddConstraint(row, {0, rows_ - 1});
  map.AddConstraint(CreateDimExpr(0, ctx) % threads_per_row_, {0, 0});
  map.Simplify();
  return map;
}

IndexingMap MetalRowNormEmitter::TileWriteMap(MLIRContext* ctx) const {
  SymbolicExpr thread = CreateDimExpr(0, ctx);
  SymbolicExpr lane = thread % threads_per_row_;
  IndexingMap map = ThreadMap(
      ctx, {thread.floorDiv(threads_per_row_), lane.floorDiv(warp_size_)}, {});
  map.AddConstraint(lane % warp_size_, {0, 0});
  return map;
}

IndexingMap MetalRowNormEmitter::TileReadMap(MLIRContext* ctx) const {
  SymbolicExpr thread = CreateDimExpr(0, ctx);
  SymbolicExpr lane = (thread % threads_per_row_) % warp_size_;
  IndexingMap map =
      ThreadMap(ctx, {thread.floorDiv(threads_per_row_), lane}, {});
  map.AddConstraint(lane, {0, threads_per_row_ / warp_size_ - 1});
  return map;
}

std::vector<emitters::EpilogueSpecification>
MetalRowNormEmitter::GetEpilogues(const HloFusionInstruction& fusion,
                                  MLIRContext* mlir_context) const {
  auto spec = [&](std::vector<const HloInstruction*> heroes,
                  std::vector<const HloInstruction*> roots,
                  absl::Span<const int64_t> dims) {
    emitters::EpilogueSpecification epilogue;
    epilogue.heroes = std::move(heroes);
    epilogue.index_ranges.assign(dims.begin(), dims.end());
    for (const HloInstruction* root : roots) {
      epilogue.roots.push_back(root);
      epilogue.root_indexing.push_back(
          GetBitcastMap(dims, root->shape(), mlir_context));
    }
    return epilogue;
  };
  std::vector<emitters::EpilogueSpecification> epilogues;
  // The input of reduction k, given reductions 0..k-1.
  for (const auto& [k, input] : input_epilogues_) {
    epilogues.push_back(spec(
        std::vector<const HloInstruction*>(reductions_.begin(),
                                           reductions_.begin() + k),
        {input}, full_dims_));
  }
  // Then the other [rows..., n] outputs and the [rows...] ones that are not
  // reductions themselves. (With all reductions as heroes, every
  // reduction's operands are roots of the partitioned computation.)
  epilogues.push_back(spec(reductions_, epilogue_full_roots_, full_dims_));
  epilogues.push_back(spec(reductions_, epilogue_row_roots_, row_dims_));
  return epilogues;
}

absl::Status MetalRowNormEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  MLIRContext* ctx = entry_function->getContext();
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  SmallVector<Value> ids = EmitThreadAndBlockIds(b);
  const emitters::PartitionedComputation& computation =
      computations.FindPartitionedComputation(
          fusion.fused_instructions_computation());
  const int num_reductions = reductions_.size();

  SmallVector<Value> outputs(
      entry_function.getArguments().drop_front(fusion.fused_parameters().size()));
  absl::flat_hash_map<const HloInstruction*, int> output_index;
  {
    int i = 0;
    for (const HloInstructionAdaptor& root : analysis_.fusion_roots()) {
      output_index[&root.instruction()] = i++;
    }
  }

  const IndexingMap loop_map = RowLoopMap(ctx);
  HloValueMap injected;
  // A reduction input at `indices`, given the reductions before it.
  auto input_value = [&](const HloInstruction* input, ValueRange indices,
                         ImplicitLocOpBuilder& nb) -> Value {
    const int epilogue = input_epilogue_.at(input);
    if (epilogue < 0) {
      return emitters::ProvideParameter(computation, reductions_.front(), 0,
                                        indices, call_targets, entry_function,
                                        nb)[0];
    }
    return emitters::EmitEpilogue(epilogue, computations, entry_function,
                                  injected, indices, nb)
        .at(input)[0];
  };
  for (int k = 0; k < num_reductions; ++k) {
    const HloInstruction* reduction = reductions_[k];
    mlir::func::FuncOp reducer =
        call_targets(reduction->called_computations()[0]->root_instruction());
    Value init = emitters::ProvideParameterRange(computation, reduction, 1, 1,
                                                 {}, call_targets,
                                                 entry_function, b)[0];
    auto body = [&](ImplicitLocOpBuilder& nb, ValueRange ivs,
                    ValueRange indices,
                    ValueRange iter_args) -> SmallVector<Value> {
      Value element = input_value(reduction->operand(0), indices, nb);
      return PureCallOp::create(nb, reducer, ValueRange{iter_args[0], element})
          .getResults();
    };
    Value partial =
        emitters::EmitXlaLoopOp(b, ids, {init}, loop_map, body)[0];
    // Lane 0 gets the simdgroup's result.
    Value value =
        ShuffleReduceOp::create(b, reducer, ValueRange{partial}, warp_size_ / 2)
            .getResult(0);
    if (threads_per_row_ > warp_size_) {
      // Lane 0 of each of the row's simdgroups writes its result to a tile
      // used by this reduction only; after the barrier the first lanes of
      // every simdgroup read them back and shuffle again (to lane 0).
      const int64_t warps_per_row = threads_per_row_ / warp_size_;
      auto tile_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          reduction->shape().element_type(), {rows_per_block_, warps_per_row});
      Value tile = AllocateSharedOp::create(
          b, emitters::TensorShapeToMlirType(tile_shape, b));
      IndexingMap write_map = TileWriteMap(ctx);
      Value write = emitters::CheckConstraints(write_map, ids, {}, b);
      tile = PredicatedInsertOp::create(
          b, write, value, tile, emitters::ApplyIndexing(write_map, ids, {}, b));
      tile = SyncThreadsOp::create(b, mlir::TypeRange{tile.getType()},
                                   ValueRange{tile})
                 .getResult(0);
      IndexingMap read_map = TileReadMap(ctx);
      Value read = emitters::CheckConstraints(read_map, ids, {}, b);
      Value partial_of_warp = PredicatedExtractOp::create(
          b, read, init, tile, emitters::ApplyIndexing(read_map, ids, {}, b));
      value = ShuffleReduceOp::create(b, reducer, ValueRange{partial_of_warp},
                                      warp_size_ / 2)
                  .getResult(0);
    }
    injected[reduction] = {BroadcastLane0(b, value, warp_size_)};
  }

  if (!full_roots_.empty()) {
    auto body = [&](ImplicitLocOpBuilder& nb, ValueRange ivs,
                    ValueRange indices,
                    ValueRange iter_args) -> SmallVector<Value> {
      absl::flat_hash_map<const HloInstruction*, ValueRange> values;
      if (!epilogue_full_roots_.empty()) {
        values = emitters::EmitEpilogue(full_epilogue_, computations,
                                        entry_function, injected, indices, nb);
      }
      SmallVector<Value> results(iter_args);
      for (const HloInstruction* root : full_roots_) {
        Value value = input_epilogue_.contains(root)
                          ? input_value(root, indices, nb)
                          : values.at(root)[0];
        int i = output_index.at(root);
        results[i] = mlir::tensor::InsertOp::create(
            nb, value, results[i],
            emitters::ApplyIndexing(
                GetBitcastMap(full_dims_, root->shape(), ctx), indices, {},
                nb));
      }
      return results;
    };
    SmallVector<Value> written =
        emitters::EmitXlaLoopOp(b, ids, outputs, loop_map, body);
    outputs.assign(written.begin(), written.end());
  }

  if (!row_roots_.empty()) {
    IndexingMap row_map = RowOutputMap(ctx);
    Value lane0 = emitters::CheckConstraints(row_map, ids, {}, b);
    SmallVector<Value> row_indices =
        emitters::ApplyIndexing(row_map, ids, {}, b);
    // The row epilogue runs in lane 0 of real rows only: it may read
    // [rows...] parameters, which padding rows (the last threadgroup's, and
    // the grid's extra blocks) would read past the end of.
    absl::flat_hash_map<const HloInstruction*, Value> values;
    if (!epilogue_row_roots_.empty()) {
      SmallVector<mlir::Type> types;
      for (const HloInstruction* root : epilogue_row_roots_) {
        types.push_back(mlir::cast<mlir::RankedTensorType>(
                            outputs[output_index.at(root)].getType())
                            .getElementType());
      }
      auto if_op = mlir::scf::IfOp::create(b, types, lane0,
                                           /*withElseRegion=*/true);
      b.setInsertionPointToStart(if_op.thenBlock());
      auto computed = emitters::EmitEpilogue(
          row_epilogue_, computations, entry_function, injected, row_indices,
          b);
      SmallVector<Value> yielded;
      for (const HloInstruction* root : epilogue_row_roots_) {
        yielded.push_back(computed.at(root)[0]);
      }
      mlir::scf::YieldOp::create(b, yielded);
      b.setInsertionPointToStart(if_op.elseBlock());
      SmallVector<Value> zeros;
      for (mlir::Type type : types) {
        zeros.push_back(mlir::arith::ConstantOp::create(
            b, type, mlir::cast<mlir::TypedAttr>(b.getZeroAttr(type))));
      }
      mlir::scf::YieldOp::create(b, zeros);
      b.setInsertionPointAfter(if_op);
      for (auto [root, result] :
           llvm::zip(epilogue_row_roots_, if_op.getResults())) {
        values[root] = result;
      }
    }
    for (const HloInstruction* root : row_roots_) {
      Value value = root->opcode() == HloOpcode::kReduce
                        ? injected.at(root)[0]
                        : values.at(root);
      int i = output_index.at(root);
      outputs[i] = PredicatedInsertOp::create(
          b, lane0, value, outputs[i],
          emitters::ApplyIndexing(GetBitcastMap(row_dims_, root->shape(), ctx),
                                  row_indices, {}, b));
    }
  }

  mlir::func::ReturnOp::create(b, outputs);
  return absl::OkStatus();
}

std::unique_ptr<FusionInterface> MetalRowNormEmitterFactory(
    const FusionInfo& fusion_info) {
  const HloFusionAnalysis& analysis = fusion_info.analysis();
  if (analysis.fusion_backend_config().custom_fusion_config().name() !=
      kMetalRowNormFusionName) {
    return nullptr;
  }
  return std::make_unique<MlirKernelFusion>(
      std::make_unique<MetalRowNormEmitter>(analysis));
}

}  // namespace gpu
}  // namespace xla
