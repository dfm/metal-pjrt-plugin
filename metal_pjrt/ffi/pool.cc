// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/ffi/pool.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/kernels/pool.metal.h"
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {
namespace {

// kernels/pool.metal's PoolParams.
struct PoolParams {
  uint32_t a, d1, d2, b;
  uint32_t k1, k2;
  uint32_t p1, p2;
  uint32_t w1;
  float init;
};

const char* TypeName(PoolType type) {
  const char* names[] = {"float", "half", "bfloat"};
  return names[static_cast<int>(type)];
}

}  // namespace

absl::StatusOr<const rt::Kernel*> GetPoolMaxBwdKernel(rt::Device* device,
                                                      PoolType type,
                                                      int lanes) {
  return device->GetKernel(
      kernels::kPoolMsl,
      absl::StrCat("pool_max_bwd_", TypeName(type), "_v", lanes));
}

absl::Status RunPoolMaxBwd(rt::Device* device, rt::Stream* stream,
                           const void* x, const void* dy, void* dx,
                           absl::Span<const int64_t> dims,
                           absl::Span<const int64_t> window, float init,
                           PoolType type) {
  const int64_t rank = dims.size();
  if (static_cast<int64_t>(window.size()) != rank || rank == 0) {
    return absl::InvalidArgumentError(
        "metal$pool_max_bwd: dims and window must have one nonzero rank");
  }
  // The windowed dims (k > 1): at most two, adjacent.
  int64_t first = -1, last = -1, elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] < 0 || window[i] < 1) {
      return absl::InvalidArgumentError(
          "metal$pool_max_bwd: negative dim or window < 1");
    }
    elements *= dims[i];
    if (window[i] > 1) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) first = last = rank - 1;  // 1 x 1 windows: dx = init + dy
  if (last - first > 1) {
    return absl::InvalidArgumentError(
        "metal$pool_max_bwd: windows on more than two adjacent dims");
  }
  if (elements == 0) return absl::OkStatus();
  if (elements > (int64_t{1} << 31) - 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Metal: max-pool backward of ", elements,
        " elements (the kernel indexes in 32 bits)"));
  }
  int64_t a = 1, b = 1;
  for (int64_t i = 0; i < first; ++i) a *= dims[i];
  for (int64_t i = last + 1; i < rank; ++i) b *= dims[i];
  const int64_t d1 = dims[first], k1 = window[first];
  const int64_t d2 = last > first ? dims[last] : 1;
  const int64_t k2 = last > first ? window[last] : 1;
  PoolParams prm;
  prm.a = static_cast<uint32_t>(a);
  prm.d1 = static_cast<uint32_t>(d1);
  prm.d2 = static_cast<uint32_t>(d2);
  prm.b = static_cast<uint32_t>(b);
  prm.k1 = static_cast<uint32_t>(k1);
  prm.k2 = static_cast<uint32_t>(k2);
  prm.p1 = static_cast<uint32_t>(d1 / k1);
  prm.p2 = static_cast<uint32_t>(d2 / k2);
  prm.w1 = static_cast<uint32_t>((d1 + k1 - 1) / k1);
  prm.init = init;
  // 4 lanes (8- or 16-byte accesses) when the unwindowed minor dims and the
  // operands' alignment allow.
  const uintptr_t vec_bytes = type == PoolType::kF32 ? 16 : 8;
  bool aligned = true;
  for (const void* ptr : {x, dy, static_cast<const void*>(dx)}) {
    aligned = aligned && reinterpret_cast<uintptr_t>(ptr) % vec_bytes == 0;
  }
  const int lanes = b % 4 == 0 && aligned ? 4 : 1;
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* kernel,
                        GetPoolMaxBwdKernel(device, type, lanes));
  const uint32_t vecs = static_cast<uint32_t>(b / lanes);
  const uint32_t tx = std::clamp<uint32_t>((vecs + 31) / 32 * 32, 32, 256);
  const rt::Dim3 threads{tx, 1, 1};
  const rt::Dim3 groups{(vecs + tx - 1) / tx,
                        static_cast<uint32_t>((d2 + k2 - 1) / k2),
                        static_cast<uint32_t>(a) * prm.w1};
  return rt::LaunchKernel(stream, *kernel, {x, dy, dx}, prm, groups, threads);
}

}  // namespace ffi
}  // namespace metal_pjrt
