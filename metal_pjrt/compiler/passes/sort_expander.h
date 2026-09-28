#ifndef METAL_PJRT_COMPILER_PASSES_SORT_EXPANDER_H_
#define METAL_PJRT_COMPILER_PASSES_SORT_EXPANDER_H_

#include <cstdint>
#include <limits>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites every kSort into a bitonic sorting network built from HLO that the
// MLIR emitters handle (XLA:GPU's own sort emitter is legacy LLVM IR, which the
// Metal backend cannot run).
//
// Each operand is transposed so the sort dimension is minor, flattened to
// [B, n], padded to [B, N] (N = next power of two) and flattened to [B*N]. An
// extra "original flat index" operand rides along. The network's
// log2(N)*(log2(N)+1)/2 substages run in a `while` loop with a known trip
// count, two elementwise compare-and-swap substages per iteration (plus one
// peeled step when the count is odd). In substage j the partner of flat
// position f is f ^ j (valid across rows because N is a power of two),
// fetched with a 1-D gather. The comparator is inlined elementwise (falling back to a
// kMap when it contains non-elementwise ops) and made a strict total order by
// breaking ties on the original index, so the result is always stable. Padded
// elements (original position >= n) compare greater than every real element.
//
// By default every sort is expanded. With a filter, only sorts whose sort
// dimension is at most `max_sort_dim` and that have more than `min_elements`
// elements: RunHloPasses uses that to keep batched tiny sorts (the
// straight-line network, <= 64 per row) away from XLA's SortRewriter, which
// takes every simple sort of more than 16384 elements and whose radix sort
// (cub_sort_ffi.cc) runs one threadgroup per row.
// Sort dimensions up to this size are expanded straight-line (every substage
// a fusion, no while loop).
inline constexpr int64_t kMaxUnrolledSortDim = 64;

class MetalSortExpander : public HloModulePass {
 public:
  MetalSortExpander() = default;
  MetalSortExpander(int64_t max_sort_dim, int64_t min_elements)
      : max_sort_dim_(max_sort_dim), min_elements_(min_elements) {}
  absl::string_view name() const override { return "metal-sort-expander"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t max_sort_dim_ = std::numeric_limits<int64_t>::max();
  int64_t min_elements_ = -1;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_SORT_EXPANDER_H_
