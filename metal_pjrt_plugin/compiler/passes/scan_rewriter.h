#ifndef METAL_PJRT_PLUGIN_COMPILER_PASSES_SCAN_REWRITER_H_
#define METAL_PJRT_PLUGIN_COMPILER_PASSES_SCAN_REWRITER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Replaces cumulative reductions over the minor dimension with a
// "metal$scan" FFI custom call (metal_pjrt_plugin/ffi/scan_ffi.cc), a single
// pass instead of the ~5 passes ReduceWindowRewriter builds.
//
// Must run before AssociativeScanRewriter / ReduceWindowRewriter (both in
// GpuCompiler's RunOptimizationPasses), so MetalCompiler runs it at the start
// of RunHloPasses. JAX lowers cumsum/cumprod/cummax/cummin to
//
//   reduce-window(X, init), window={size=1x..xn, pad=0_0x..x(n-1)_0}
//
// (padding 0_(n-1) on the scanned dim when reverse=True). Matched: one
// operand, window size n = the minor dimension with low/high padding
// (n-1, 0) or (0, n-1), size 1 / no padding elsewhere, unit strides and
// dilations, reducer add/multiply/maximum/minimum of the two parameters,
// init equal to that reducer's identity, element type f32/f16/bf16/s32, a
// default (or absent) layout, n >= kMinRowLength, and (rows >=
// kMinRowsForLongRows or n <= kLongRowLength) since the kernel runs one
// threadgroup per row. Everything else
// (logcumsumexp, non-minor axes, other types) is left alone.
class MetalScanRewriter : public HloModulePass {
 public:
  static constexpr int64_t kMinRowLength = 2;
  static constexpr int64_t kLongRowLength = 4096;
  static constexpr int64_t kMinRowsForLongRows = 32;

  absl::string_view name() const override { return "metal-scan-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_PLUGIN_COMPILER_PASSES_SCAN_REWRITER_H_
