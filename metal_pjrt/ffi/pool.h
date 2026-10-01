// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Max-pool backward over non-overlapping windows on the GPU: the dispatch
// behind "metal$pool_max_bwd" (pool_ffi.cc, MetalPoolMaxBwdRewriter), with
// no XLA. dx = select-and-scatter(x, dy, init) for a GE select, an add
// scatter and windows equal to their strides (kernels/pool.metal has the
// exact selection rule).
#ifndef METAL_PJRT_FFI_POOL_H_
#define METAL_PJRT_FFI_POOL_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {

enum class PoolType { kF32, kF16, kBF16 };

// The kernel of kernels/pool.metal for this element type, `lanes` (1 or 4)
// consecutive elements per thread.
absl::StatusOr<const rt::Kernel*> GetPoolMaxBwdKernel(rt::Device* device,
                                                      PoolType type,
                                                      int lanes);

// Enqueues dx (x's shape) from x and dy (x's dims / window, rounded down)
// on the stream. `dims` and `window` have the same rank; every window is
// >= 1 and at most two adjacent dims have one > 1. InvalidArgument for
// other shapes, Unimplemented past 2^31 elements (32-bit indexing).
// Nothing is launched for an empty x.
absl::Status RunPoolMaxBwd(rt::Device* device, rt::Stream* stream,
                           const void* x, const void* dy, void* dx,
                           absl::Span<const int64_t> dims,
                           absl::Span<const int64_t> window, float init,
                           PoolType type);

}  // namespace ffi
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFI_POOL_H_
