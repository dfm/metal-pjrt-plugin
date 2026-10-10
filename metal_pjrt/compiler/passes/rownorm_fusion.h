// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_ROWNORM_FUSION_H_
#define METAL_PJRT_COMPILER_PASSES_ROWNORM_FUSION_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// custom_fusion_config name of the fusions MetalRowNormFusion builds.
inline constexpr absl::string_view kMetalRowNormFusionName = "metal_rownorm";

// True for a fusion built by MetalRowNormFusion.
bool IsMetalRowNormFusion(const HloInstruction& instr);

// Fuses row normalizations (softmax, log-softmax, layer and RMS norm, and
// their gradients' row reductions) into one kernel each.
//
// XLA splits these into a reduction kernel per row reduction plus a loop
// kernel for the result: softmax reads its input three times and writes it
// once. Priority fusion cannot merge them (a reduce fusion cannot take
// another reduction, or a consumer that reads the input again). CUDA fuses
// them with SoftmaxRewriterTriton and Triton; this pass builds the same
// kind of fusion for MetalRowNormEmitter (codegen/rownorm_emitter.h), which
// gives each row to one threadgroup or simdgroup: every reduction of a row
// reads the row (from cache after the first), and the result is written
// once.
//
// A fusion grows from a reduce over the minor dimension of a rank >= 2
// array, [rows..., n] -> [rows...], with a constant init and a scalar add,
// max, min or multiply (row length n <= max_row_length). It takes in, as
// long as all are in the same row space:
//  - elementwise ops on [rows..., n] or [rows...] values,
//  - further minor-dimension reductions of its [rows..., n] values, and
//    sibling reductions of the same inputs (mean and mean of squares),
//  - broadcasts of [rows...] values along the row, and of scalars or
//    [n] vectors (layer-norm weights),
//  - producers used only inside it, and cheap ones (constants, iotas,
//    broadcasts of scalars, and elementwise ops of those: causal masks)
//    duplicated from outside.
// Its outputs are the values used outside, [rows..., n] or [rows...]: under
// JAX's autodiff of softmax the row sum and exp(x - max) are outputs too.
// A fusion is only built when some [rows..., n] value depends on a row
// reduction (a reduction alone stays with XLA's reduction emitter).
//
// Must run after layout normalization (every value in a fusion has the
// default layout) and before priority fusion, which leaves custom fusions
// alone. Runs at the end of MetalCompiler::OptimizeHloPostLayoutAssignment.
class MetalRowNormFusion : public HloModulePass {
 public:
  static constexpr int64_t kMaxRowLength = 16384;

  explicit MetalRowNormFusion(int64_t max_row_length = kMaxRowLength)
      : max_row_length_(max_row_length) {}

  absl::string_view name() const override { return "metal-rownorm-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t max_row_length_;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_ROWNORM_FUSION_H_
