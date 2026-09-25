#ifndef METAL_PJRT_PLUGIN_COMPILER_PASSES_SOFTMAX_REWRITER_H_
#define METAL_PJRT_PLUGIN_COMPILER_PASSES_SOFTMAX_REWRITER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Replaces row softmax / log-softmax over the minor dimension with a
// "metal$softmax" FFI custom call (metal_pjrt_plugin/ffi/softmax_ffi.cc), one
// kernel reading the row twice instead of XLA's three kernels.
//
// Runs after layout normalization (so every array is row-major and the
// reductions are over the minor dimension) and before fusion. Matched, with
// T in {f32, f16, bf16} and cvt(.) an optional chain of float converts:
//
//   M = reduce(X, -inf), dims={last}, to_apply=max
//   D = subtract(X, broadcast(cvt(M)))
//   E = exp(D)
//   S = reduce(cvt(E), 0), dims={last}, to_apply=add
//   softmax:     root = divide(E, broadcast(cvt(S)))
//   log-softmax: root = subtract(D, broadcast(cvt(log(cvt(S)))))
//
// where the broadcasts map the reduced shape back onto dims {0..rank-2}, X
// and root have type T and the default layout, the minor dimension is at
// most kMaxRowLength, and every matched intermediate is used only inside the
// pattern (otherwise the old kernels would still run and nothing is saved).
class MetalSoftmaxRewriter : public HloModulePass {
 public:
  static constexpr int64_t kMaxRowLength = 16384;

  absl::string_view name() const override { return "metal-softmax-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_PLUGIN_COMPILER_PASSES_SOFTMAX_REWRITER_H_
