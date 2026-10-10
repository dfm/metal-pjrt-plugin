// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Inclusive scan (cumsum/cumprod/cummax/cummin) over the rows of a row-major
// [rows, n] array on the GPU: the dispatch behind "metal$scan"
// (scan_ffi.cc), with no XLA, so scan_test runs it against rt::Device alone.
//
// One threadgroup per row. Each iteration covers a chunk of 4 * threads
// elements: every thread scans 4 consecutive elements in registers, the
// per-thread totals are scanned within the simdgroup (shuffles), simdgroup
// totals are scanned through threadgroup memory, and a running carry links
// the chunks. f16/bf16 accumulate in f32; s32 add/mul wrap; max/min
// propagate NaN (kernels/scan.metal, one instantiation per type and op).
#ifndef METAL_PJRT_FFI_SCAN_H_
#define METAL_PJRT_FFI_SCAN_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {

enum class ScanOp { kAdd, kMul, kMax, kMin };
enum class ScanType { kF32, kF16, kBF16, kS32 };

// "add" | "mul" | "max" | "min".
absl::StatusOr<ScanOp> ParseScanOp(absl::string_view op);

// The kernel of kernels/scan.metal for this element type and op.
std::string ScanFunction(ScanType type, ScanOp op);

// That kernel, from the device's kernel cache (compiled on first use).
absl::StatusOr<const rt::Kernel*> GetScanKernel(rt::Device* device,
                                                ScanType type, ScanOp op);

// Enqueues y[r, :] = scan(x[r, :]) for r < rows on the stream (reverse:
// scanned from the end of the row). `kernel` is GetScanKernel for the
// stream's device, or null to look it up. Nothing is launched when
// rows or n is 0; rows >= 2^32 or n > 2^31 is Unimplemented.
absl::Status RunScan(rt::Device* device, rt::Stream* stream, const void* x,
                     void* y, uint64_t rows, uint64_t n, ScanType type,
                     ScanOp op, bool reverse,
                     const rt::Kernel* kernel = nullptr);

}  // namespace ffi
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFI_SCAN_H_
