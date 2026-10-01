// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/stream_executor/metal_kernel.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "absl/status/status_macros.h"
#include "metal_pjrt/stream_executor/metal_executor.h"
#include "metal_pjrt/stream_executor/metal_stream.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

namespace {
// Metal's launch dimensions are 32-bit; StreamExecutor's are 64-bit.
absl::StatusOr<rt::Dim3> ToDim3(absl::string_view kernel,
                                absl::string_view what, uint64_t x,
                                uint64_t y, uint64_t z) {
  constexpr uint64_t kMax = std::numeric_limits<uint32_t>::max();
  if (x > kMax || y > kMax || z > kMax) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s: %s %d x %d x %d exceed Metal's 32-bit launch dimensions",
        kernel, what, x, y, z));
  }
  return rt::Dim3{static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                  static_cast<uint32_t>(z)};
}

absl::StatusOr<uint32_t> SharedBytes(absl::string_view kernel,
                                     uint64_t bytes) {
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s: %d bytes of threadgroup memory requested", kernel,
        bytes));
  }
  return static_cast<uint32_t>(bytes);
}
}  // namespace

absl::StatusOr<int32_t> MetalKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  uint64_t per_block = threads.x * threads.y * threads.z;
  if (per_block == 0) return 1;
  int32_t blocks =
      static_cast<int32_t>(kernel_->max_total_threads_per_threadgroup() /
                           per_block);
  return blocks < 1 ? 1 : blocks;
}

absl::Status MetalKernel::CheckClusterDims(
    absl::string_view name, const std::optional<ClusterDim>& cluster_dims) {
  if (cluster_dims.has_value() &&
      (cluster_dims->x != 1 || cluster_dims->y != 1 || cluster_dims->z != 1)) {
    return absl::UnimplementedError(absl::StrFormat(
        "kernel %s: thread block clusters (%d x %d x %d) are not supported on "
        "Metal",
        name, cluster_dims->x, cluster_dims->y, cluster_dims->z));
  }
  return absl::OkStatus();
}

absl::Status MetalKernel::PackArgs(
    const KernelArgs& args, absl::InlinedVector<rt::KernelArg, 16>& rt_args,
    uint32_t& shared_bytes) const {
  rt_args.clear();
  ABSL_ASSIGN_OR_RETURN(shared_bytes,
                        SharedBytes(name(), args.number_of_shared_bytes()));

  if (auto* device_args = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    rt_args.reserve(device_args->device_addr_args().size());
    for (const DeviceAddressBase& mem : device_args->device_addr_args()) {
      rt_args.push_back(rt::KernelArg::Buffer(mem.opaque()));
    }
  } else if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    // Re-pack when a packing function is registered and the args allow it.
    std::unique_ptr<KernelArgsPackedArrayBase> repacked;
    const KernelArgsPackedArrayBase* use = packed;
    auto& pack = args_packing();
    if (pack && dynamic_cast<const PackableKernelArgs*>(&args) != nullptr) {
      ABSL_ASSIGN_OR_RETURN(repacked, pack(*this, args));
      use = repacked.get();
    }
    absl::Span<const void* const> addrs = use->argument_addresses();
    rt_args.reserve(addrs.size());
    for (const void* const arg : addrs) {
      // Each argument is the address of an 8-byte device pointer.
      rt_args.push_back(rt::KernelArg::Buffer(
          *static_cast<void* const*>(arg)));
    }
    ABSL_ASSIGN_OR_RETURN(shared_bytes,
                          SharedBytes(name(), use->number_of_shared_bytes()));
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s: unsupported KernelArgs kind (expected device address "
        "array or packed arguments)",
        name()));
  }

  if (rt_args.size() != arity_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s expects %u arguments, got %u", name(), arity_,
        rt_args.size()));
  }
  return absl::OkStatus();
}

absl::Status MetalKernel::Launch(const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const std::optional<ClusterDim>& cluster_dims,
                                 Stream* stream, const KernelArgs& args) {
  ABSL_RETURN_IF_ERROR(CheckClusterDims(name(), cluster_dims));
  absl::InlinedVector<rt::KernelArg, 16> rt_args;
  uint32_t shared_bytes = 0;
  ABSL_RETURN_IF_ERROR(PackArgs(args, rt_args, shared_bytes));

  auto* metal_stream = static_cast<MetalStream*>(stream);
  ABSL_ASSIGN_OR_RETURN(rt::Dim3 groups,
                        ToDim3(name(), "threadgroups", block_dims.x,
                               block_dims.y, block_dims.z));
  ABSL_ASSIGN_OR_RETURN(rt::Dim3 threads,
                        ToDim3(name(), "threads per threadgroup",
                               thread_dims.x, thread_dims.y, thread_dims.z));
  return metal_stream->rt_stream()->Launch(*kernel_, groups, threads, rt_args,
                                           shared_bytes);
}

}  // namespace metal
}  // namespace stream_executor
