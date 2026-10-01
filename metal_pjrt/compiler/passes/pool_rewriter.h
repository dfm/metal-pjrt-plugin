// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_POOL_REWRITER_H_
#define METAL_PJRT_COMPILER_PASSES_POOL_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Replaces the gradient of a max pool over non-overlapping windows,
//
//   select-and-scatter(x, dy, init), select=GE, scatter=add,
//
// with a "metal$pool_max_bwd" FFI custom call (metal_pjrt/ffi/pool_ffi.cc):
// one pass writing dx, instead of SelectAndScatterExpander's reduce-window
// over x and one s32 iota per dimension (the max and its index arrays)
// followed by an atomic scatter-add into a filled dx.
//
// Matched conservatively: f32/f16/bf16, windows > 1 on at most two adjacent
// dimensions, every window dimension with size == stride, no padding, no
// dilation, no reversal; the select
// computation exactly compare(p0, p1), direction=GE; the scatter exactly
// add(p0, p1); a constant scalar init; a default (or absent) layout; fewer
// than 2^31 elements; at most 64 elements per window. The kernel selects
// exactly as the expander does (first maximum; NaN as `>=` orders it).
// Anything not matched is left alone (never refused).
// Everything else (overlapping or padded windows, min pools, other
// selects) stays with the expander. Runs at the start of RunHloPasses,
// before GpuCompiler's SelectAndScatterExpander.
class MetalPoolMaxBwdRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "metal-pool-max-bwd-rewriter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_POOL_REWRITER_H_
