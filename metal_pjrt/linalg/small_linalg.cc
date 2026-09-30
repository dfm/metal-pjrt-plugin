// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/linalg/small_linalg.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/kernels/small_linalg.metal.h"
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace linalg {
namespace {

// SmallParams::flags bits.
constexpr uint32_t kFlagLower = 1;
constexpr uint32_t kFlagLeft = 2;
constexpr uint32_t kFlagTrans = 4;
constexpr uint32_t kFlagUnit = 8;

// Matches the MSL struct of the same name.
struct SmallParams {
  uint32_t batch;
  uint32_t m;      // rows of the matrix being factorized / of B
  uint32_t n;      // cols of the matrix being factorized / of B
  uint32_t k;      // order of the triangular matrix
  uint32_t flags;
};

// The small kernels compute offsets (b * n * n, gid) in 32-bit uint: at 2^32
// elements or threads they would wrap and silently alias other batch
// elements, so such calls take the host path instead.
bool FitsSmallKernel(std::initializer_list<uint64_t> counts) {
  for (uint64_t c : counts) {
    if (c > 0xffffffffu) return false;
  }
  return true;
}

absl::Status LaunchSmall(rt::Device* device, rt::Stream* stream,
                         const char* function,
                         const std::vector<const void*>& buffers,
                         const SmallParams& params, uint64_t threads) {
  // An empty batch has nothing to do and may have null buffers to bind.
  // (XLA folds zero-element ops away before this, so it is a backstop.)
  if (threads == 0) return absl::OkStatus();
  const uint32_t group = 64;
  const uint32_t groups =
      static_cast<uint32_t>((threads + group - 1) / group);
  absl::StatusOr<const rt::Kernel*> kernel =
      device->GetKernel(kernels::kSmallLinalgMsl, function);
  if (!kernel.ok()) return kernel.status();
  return rt::LaunchKernel(stream, **kernel, buffers, params,
                          rt::Dim3{std::max(groups, 1u), 1, 1},
                          rt::Dim3{group, 1, 1});
}

}  // namespace

bool UseSmallCholesky(int64_t batch, int64_t n) {
  return n <= kSmallMatrixMax && n > 0 &&
         FitsSmallKernel({static_cast<uint64_t>(batch * n * n)});
}

bool UseSmallTrsm(int64_t batch, int64_t m, int64_t n, bool left_side) {
  const int64_t k = left_side ? m : n;
  const uint64_t lines = left_side ? n : m;
  return k <= kSmallMatrixMax && k > 0 && m > 0 && n > 0 &&
         FitsSmallKernel({static_cast<uint64_t>(batch * k * k),
                          static_cast<uint64_t>(batch * m * n),
                          static_cast<uint64_t>(batch) * lines});
}

bool UseSmallGetrf(int64_t batch, int64_t m, int64_t n) {
  return m <= kSmallMatrixMax && n <= kSmallMatrixMax && m > 0 && n > 0 &&
         FitsSmallKernel({static_cast<uint64_t>(batch * m * n)});
}

absl::Status RunSmallCholesky(rt::Device* device, rt::Stream* stream,
                              const float* a, float* out, int64_t batch,
                              int64_t n, bool lower) {
  SmallParams p{static_cast<uint32_t>(batch), 0, static_cast<uint32_t>(n), 0,
                lower ? kFlagLower : 0u};
  return LaunchSmall(device, stream, "small_cholesky", {a, out}, p, batch);
}

absl::Status RunSmallTrsm(rt::Device* device, rt::Stream* stream,
                          const float* a, const float* b, float* x,
                          int64_t batch, int64_t m, int64_t n, bool left_side,
                          bool lower, bool transpose, bool unit_diagonal) {
  const int64_t k = left_side ? m : n;
  const uint64_t lines = left_side ? n : m;
  SmallParams p{static_cast<uint32_t>(batch), static_cast<uint32_t>(m),
                static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                (lower ? kFlagLower : 0u) | (left_side ? kFlagLeft : 0u) |
                    (transpose ? kFlagTrans : 0u) |
                    (unit_diagonal ? kFlagUnit : 0u)};
  return LaunchSmall(device, stream, "small_trsm", {a, b, x}, p,
                     batch * lines);
}

absl::Status RunSmallGetrf(rt::Device* device, rt::Stream* stream,
                           const float* a, float* lu, int32_t* pivots,
                           int32_t* permutation, int64_t batch, int64_t m,
                           int64_t n) {
  SmallParams p{static_cast<uint32_t>(batch), static_cast<uint32_t>(m),
                static_cast<uint32_t>(n),
                static_cast<uint32_t>(std::min(m, n)), 0};
  return LaunchSmall(device, stream, "small_getrf", {a, lu, pivots, permutation},
                     p, batch);
}

}  // namespace linalg
}  // namespace metal_pjrt
