// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/ffi/radix_sort.h"

#include <cstdint>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/kernels/cub_sort.metal.h"
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint64_t kTile = 2048;  // kThreads * kItems in the MSL
constexpr int kDigitBits = 4;

struct Params {
  uint32_t n;
  uint32_t tiles;
  uint32_t shift;
  uint32_t descending;
};

bool ValidBits(int bits) {
  return bits == 8 || bits == 16 || bits == 32 || bits == 64;
}

// Key and value bits' storage type as named in the kernels' instantiations
// (kernels/cub_sort.metal).
const char* BitsType(int bits) {
  switch (bits) {
    case 8: return "u8";
    case 16: return "u16";
    case 64: return "u64";
    default: return "u32";
  }
}

uint64_t Align256(uint64_t x) { return (x + 255) / 256 * 256; }

}  // namespace

absl::StatusOr<RadixSortPlan> PlanRadixSort(const RadixSortSpec& spec) {
  if (!ValidBits(spec.key_bits) ||
      (spec.value_bits != 0 && !ValidBits(spec.value_bits))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal radix sort: unsupported key/value widths ",
                     spec.key_bits, "/", spec.value_bits));
  }
  RadixSortPlan plan;
  plan.key_bits = spec.key_bits;
  plan.key_kind = spec.key_kind;
  plan.value_bits = spec.value_bits;
  plan.key_bytes = spec.key_bits / 8;
  plan.value_bytes = spec.value_bits / 8;
  plan.total = spec.total;
  if (plan.total == 0) return plan;
  const int64_t batch_size = spec.batch_size;
  if (batch_size <= 0 || plan.total % batch_size != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal radix sort: batch_size ", batch_size, " does not divide ",
        plan.total, " elements"));
  }
  // 32-bit offsets and counts in the kernels.
  if (plan.total > (uint64_t{1} << 31)) {
    return absl::UnimplementedError(
        "Metal radix sort: more than 2^31 elements");
  }
  plan.batch = batch_size;
  plan.n = plan.total / plan.batch;
  plan.small = plan.n <= kTile;
  plan.tiles = (plan.n + kTile - 1) / kTile;
  if (!plan.small) {
    plan.alt_values_offset = Align256(plan.total * plan.key_bytes);
    plan.hist_offset =
        plan.alt_values_offset +
        (plan.value_bits != 0 ? Align256(plan.total * plan.value_bytes) : 0);
    plan.scratch_bytes = plan.hist_offset + Align256(plan.batch * plan.tiles *
                                                     16 * sizeof(uint32_t));
  }
  return plan;
}

absl::StatusOr<RadixSortKernels> GetRadixSortKernels(
    rt::Device* device, const RadixSortPlan& plan) {
  const bool has_values = plan.value_bits != 0;
  // The function_constant indices of kernels/cub_sort.metal.
  const rt::FunctionConstant constants[] = {
      rt::FunctionConstant::Int(0, static_cast<int>(plan.key_kind)),
      rt::FunctionConstant::Bool(1, has_values)};
  const char* key_type = BitsType(plan.key_bits);
  // Keys only: any instantiation (the kernels never touch the values).
  const char* value_type = has_values ? BitsType(plan.value_bits) : "u32";
  const std::string kv = absl::StrCat(key_type, "_", value_type);
  RadixSortKernels k;
  for (auto [out, name, with_constants] :
       {std::make_tuple(&k.small, absl::StrCat("sort_small_", kv), true),
        std::make_tuple(&k.hist, absl::StrCat("sort_hist_", key_type), true),
        std::make_tuple(&k.scan, std::string("sort_scan"), false),
        std::make_tuple(&k.scatter, absl::StrCat("sort_scatter_", kv),
                        true)}) {
    absl::StatusOr<const rt::Kernel*> kernel = device->GetKernel(
        kernels::kCubSortMsl, name,
        with_constants ? absl::Span<const rt::FunctionConstant>(constants)
                       : absl::Span<const rt::FunctionConstant>());
    if (!kernel.ok()) return kernel.status();
    // The kernels assume exactly kThreads threads in 32-wide SIMD groups.
    if ((*kernel)->max_total_threads_per_threadgroup() < kThreads ||
        (*kernel)->thread_execution_width() != 32) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal radix sort: ", name, " cannot run 256 threads in 32-wide "
          "SIMD groups on this device"));
    }
    *out = *kernel;
  }
  return k;
}

absl::Status RunRadixSort(rt::Device* device, rt::Stream* stream,
                          const RadixSortPlan& plan, const void* keys_in,
                          void* keys_out, const void* values_in,
                          void* values_out, void* scratch,
                          uint64_t scratch_bytes, bool descending) {
  // Refuse before encoding: the kernels would write past a short scratch.
  if (plan.scratch_bytes > scratch_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal radix sort: scratch buffer has ", scratch_bytes,
        " bytes, needs ", plan.scratch_bytes));
  }
  if (plan.total == 0) return absl::OkStatus();
  absl::StatusOr<RadixSortKernels> k = GetRadixSortKernels(device, plan);
  if (!k.ok()) return k.status();

  const bool has_values = plan.value_bits != 0;
  const void* kin = keys_in;
  void* kout = keys_out;
  // Keys-only kernels never touch the value buffers; bind the keys instead.
  const void* vin = has_values ? values_in : kin;
  void* vout = has_values ? values_out : kout;
  Params p{static_cast<uint32_t>(plan.n), static_cast<uint32_t>(plan.tiles),
           0, descending ? 1u : 0u};
  const rt::Dim3 threads{kThreads, 1, 1};
  if (plan.small) {
    return rt::LaunchKernel(stream, *k->small, {kin, kout, vin, vout}, p,
                            rt::Dim3{static_cast<uint32_t>(plan.batch), 1, 1},
                            threads);
  }
  auto* base = static_cast<uint8_t*>(scratch);
  void* alt_keys = base;
  void* alt_vals = has_values ? base + plan.alt_values_offset : alt_keys;
  void* hist = base + plan.hist_offset;
  const rt::Dim3 tiles{static_cast<uint32_t>(plan.batch * plan.tiles), 1, 1};
  const int passes = plan.key_bits / kDigitBits;  // even: 2, 4, 8 or 16
  const void* src_k = kin;
  const void* src_v = vin;
  for (int pass = 0; pass < passes; ++pass) {
    // The last pass lands in the outputs; so does every other one before it.
    const bool to_out = (passes - 1 - pass) % 2 == 0;
    void* dst_k = to_out ? kout : alt_keys;
    void* dst_v = to_out ? vout : alt_vals;
    p.shift = pass * kDigitBits;
    if (absl::Status s = rt::LaunchKernel(stream, *k->hist, {src_k, hist}, p,
                                          tiles, threads);
        !s.ok()) {
      return s;
    }
    if (absl::Status s = rt::LaunchKernel(
            stream, *k->scan, {hist}, p,
            rt::Dim3{static_cast<uint32_t>(plan.batch), 1, 1}, threads);
        !s.ok()) {
      return s;
    }
    if (absl::Status s = rt::LaunchKernel(stream, *k->scatter,
                                          {src_k, dst_k, src_v, dst_v, hist},
                                          p, tiles, threads);
        !s.ok()) {
      return s;
    }
    src_k = dst_k;
    src_v = dst_v;
  }
  return absl::OkStatus();
}

}  // namespace ffi
}  // namespace metal_pjrt
