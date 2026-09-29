// An LSD radix sort of the rows of [batch, n] keys (and optional values) on
// the GPU: the dispatch behind XLA's SortRewriter targets
// "xla.gpu.ext.cub_sort_keys/pairs" (cub_sort_ffi.cc), with no XLA, so
// radix_sort_test runs it against rt::Device alone.
//
// Each row is sorted independently, stably in both directions (equal keys
// keep their input order), by key bits as CUB orders them: unsigned as is,
// signed with the sign bit flipped, floats in total order (negative: all
// bits flipped, positive: sign flipped) except that -0 sorts as +0. Values
// are raw bits.
//
// 4-bit digits (16 buckets), 256 threads x 8 items per tile. Every pass ranks
// a tile stably in threadgroup memory: per-thread digit counts laid out
// [digit][thread], one exclusive scan, rank = scanned count + position among
// the thread's own items. Two paths:
//  - rows of at most one tile (2048): one threadgroup per row runs all the
//    passes in threadgroup memory (sort_small), no scratch;
//  - longer rows: per pass, sort_hist (per-tile digit counts, laid out
//    [row][digit][tile]), sort_scan (exclusive scan per row) and sort_scatter
//    (each item to its row offset), ping-ponging between the outputs and
//    scratch copies of keys and values; the input is never written. Scratch:
//    [keys | values | counts], each 256-byte aligned.
//
// GPU safety: every kernel loop is bounded by the key width or the sizes in
// Params; scatter destinations are checked against the row length;
// RunRadixSort refuses, before encoding anything, a scratch buffer smaller
// than the plan's.
#ifndef METAL_PJRT_FFI_RADIX_SORT_H_
#define METAL_PJRT_FFI_RADIX_SORT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {

// How key bits order (the KIND function constant of kernels/cub_sort.metal).
enum class KeyKind { kUnsigned = 0, kSigned = 1, kFloat = 2 };

struct RadixSortSpec {
  int key_bits = 32;  // 8, 16, 32 or 64
  KeyKind key_kind = KeyKind::kUnsigned;
  int value_bits = 0;  // 8, 16, 32 or 64; 0 sorts keys only
  uint64_t total = 0;  // elements (batch_size rows)
  int64_t batch_size = 1;
};

// Everything the sort derives from the spec: computed the same way for the
// scratch estimate and before running, to check the scratch buffer.
struct RadixSortPlan {
  int key_bits = 32;
  KeyKind key_kind = KeyKind::kUnsigned;
  int value_bits = 0;
  uint64_t key_bytes = 0, value_bytes = 0;
  uint64_t total = 0, batch = 0, n = 0, tiles = 0;
  bool small = true;
  // Scratch layout (large path); scratch_bytes is 0 on the small path.
  uint64_t alt_values_offset = 0, hist_offset = 0, scratch_bytes = 0;
};

// Fails if batch_size does not divide a nonzero total, or above 2^31
// elements (32-bit offsets and counts in the kernels). An empty sort plans
// nothing.
absl::StatusOr<RadixSortPlan> PlanRadixSort(const RadixSortSpec& spec);

struct RadixSortKernels {
  const rt::Kernel* small = nullptr;
  const rt::Kernel* hist = nullptr;
  const rt::Kernel* scan = nullptr;
  const rt::Kernel* scatter = nullptr;
};

// From the device's kernel cache (compiled on first use). Unimplemented if
// the device cannot run them as written (256 threads, 32-wide SIMD groups).
absl::StatusOr<RadixSortKernels> GetRadixSortKernels(
    rt::Device* device, const RadixSortPlan& plan);

// Enqueues the sort of keys_in (and values_in, when the plan has values)
// into keys_out (values_out) on the stream. The inputs are not written.
// `scratch` must hold plan.scratch_bytes (InvalidArgument otherwise, before
// anything is encoded). An empty plan launches nothing.
absl::Status RunRadixSort(rt::Device* device, rt::Stream* stream,
                          const RadixSortPlan& plan, const void* keys_in,
                          void* keys_out, const void* values_in,
                          void* values_out, void* scratch,
                          uint64_t scratch_bytes, bool descending);

}  // namespace ffi
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFI_RADIX_SORT_H_
