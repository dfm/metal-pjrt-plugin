// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_CODEGEN_ROWNORM_EMITTER_H_
#define METAL_PJRT_CODEGEN_ROWNORM_EMITTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla {
namespace gpu {

// Emits a MetalRowNormFusion (compiler/passes/rownorm_fusion.h): rows of
// [rows..., n] values, each given to threads_per_row threads (one or more
// simdgroups) of a threadgroup.
//
// For each row reduction, in order, the row's threads loop over the row,
// evaluating the reduction's input (through XLA's epilogue mechanism, with
// the row's earlier reduction results injected as values), combine their
// partial results with simdgroup shuffles and, for rows of several
// simdgroups, through a threadgroup-memory tile of its own (so no tile is
// written twice and no barrier guards a reuse), and end with the row's
// result in every thread (lane 0's, shuffled to the others). A last loop writes the [rows..., n] outputs;
// lane 0 writes the [rows...] ones. Each loop recomputes the row's inputs,
// which the earlier loops brought into cache, so the row is read from
// memory about once.
class MetalRowNormEmitter : public MlirKernelEmitter {
 public:
  explicit MetalRowNormEmitter(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    return std::nullopt;
  }
  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    return std::nullopt;
  }

 protected:
  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  // (rows..., col) for the s0-th element of a thread's row, col < n.
  IndexingMap RowLoopMap(mlir::MLIRContext* ctx) const;
  // (rows...) of a thread's row, for lane 0 only.
  IndexingMap RowOutputMap(mlir::MLIRContext* ctx) const;
  // A [rows_per_block, warps_per_row] tile: where lane 0 of each simdgroup
  // writes, and where lane l < warps_per_row reads.
  IndexingMap TileWriteMap(mlir::MLIRContext* ctx) const;
  IndexingMap TileReadMap(mlir::MLIRContext* ctx) const;
  IndexingMap ThreadMap(mlir::MLIRContext* ctx,
                        llvm::ArrayRef<SymbolicExpr> results,
                        absl::Span<const int64_t> symbol_sizes) const;
  SymbolicExpr RowIndex(mlir::MLIRContext* ctx) const;

  const HloFusionAnalysis& analysis_;
  // The fusion's reductions, in post order.
  std::vector<const HloInstruction*> reductions_;
  // Fusion roots by kind (a root may be a reduction).
  std::vector<const HloInstruction*> full_roots_;
  std::vector<const HloInstruction*> row_roots_;
  // How each reduction input is evaluated: -1 through the partitioned
  // computation (the first reduction's input, which no reduction feeds),
  // else the index of its epilogue. Each distinct input gets one epilogue:
  // epilogue functions are named after their roots, so two with the same
  // roots would clash. [rows..., n] roots that are reduction inputs are
  // evaluated the same way.
  absl::flat_hash_map<const HloInstruction*, int> input_epilogue_;
  // (first reduction reading it, input), one per input epilogue.
  std::vector<std::pair<int, const HloInstruction*>> input_epilogues_;
  // The other roots, computed by the last two epilogues.
  std::vector<const HloInstruction*> epilogue_full_roots_;
  std::vector<const HloInstruction*> epilogue_row_roots_;
  int full_epilogue_ = 0;
  int row_epilogue_ = 0;
  std::vector<int64_t> full_dims_;
  std::vector<int64_t> row_dims_;
  int64_t rows_ = 0;
  int64_t n_ = 0;
  int64_t warp_size_ = 32;
  int64_t threads_per_row_ = 0;
  int64_t rows_per_block_ = 0;
  int64_t num_blocks_ = 0;
  std::array<uint64_t, 2> gpu_blocks_ = {0, 0};
};

// For SetCustomFusionEmitterFactory (XLA patch 0004): the emitter of
// metal_rownorm fusions, null for any other custom fusion.
std::unique_ptr<FusionInterface> MetalRowNormEmitterFactory(
    const FusionInfo& fusion_info);

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_CODEGEN_ROWNORM_EMITTER_H_
