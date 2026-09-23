#include "metal_pjrt_plugin/stream_executor/metal_kernel.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"
#include "metal_pjrt_plugin/stream_executor/metal_stream.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

MetalKernel::MetalKernel(MetalExecutor* executor,
                         std::unique_ptr<rt::Kernel> kernel, unsigned arity)
    : executor_(executor), kernel_(std::move(kernel)), arity_(arity) {}

MetalKernel::~MetalKernel() { executor_->UnloadKernel(this); }

absl::StatusOr<int32_t> MetalKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  uint64_t per_block = threads.x * threads.y * threads.z;
  if (per_block == 0) return 1;
  int32_t blocks =
      static_cast<int32_t>(kernel_->max_total_threads_per_threadgroup() /
                           per_block);
  return blocks < 1 ? 1 : blocks;
}

absl::Status MetalKernel::Launch(const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const std::optional<ClusterDim>& cluster_dims,
                                 Stream* stream, const KernelArgs& args) {
  if (cluster_dims.has_value() &&
      (cluster_dims->x != 1 || cluster_dims->y != 1 || cluster_dims->z != 1)) {
    return absl::UnimplementedError("Metal kernels do not support clusters");
  }

  std::vector<rt::KernelArg> rt_args;
  uint64_t shared_bytes = args.number_of_shared_bytes();

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
      TF_ASSIGN_OR_RETURN(repacked, pack(*this, args));
      use = repacked.get();
    }
    absl::Span<const void* const> addrs = use->argument_addresses();
    rt_args.reserve(addrs.size());
    for (const void* const arg : addrs) {
      // Each argument is the address of an 8-byte device pointer.
      rt_args.push_back(rt::KernelArg::Buffer(
          *static_cast<void* const*>(arg)));
    }
    shared_bytes = use->number_of_shared_bytes();
  } else {
    return absl::InvalidArgumentError("unsupported KernelArgs kind");
  }

  if (rt_args.size() != arity_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s expects %u arguments, got %u", name(), arity_,
        rt_args.size()));
  }
  if (rt_args.size() > 31) {
    return absl::UnimplementedError(absl::StrFormat(
        "kernel %s has %u buffer arguments; Metal allows at most 31 (argument "
        "buffers not implemented yet)",
        name(), rt_args.size()));
  }

  uint64_t threads_per_group = thread_dims.x * thread_dims.y * thread_dims.z;
  if (threads_per_group > kernel_->max_total_threads_per_threadgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "kernel %s: %u threads per threadgroup exceeds pipeline limit %u",
        name(), threads_per_group,
        kernel_->max_total_threads_per_threadgroup()));
  }

  auto* metal_stream = static_cast<MetalStream*>(stream);
  rt::Dim3 groups{static_cast<uint32_t>(block_dims.x),
                  static_cast<uint32_t>(block_dims.y),
                  static_cast<uint32_t>(block_dims.z)};
  rt::Dim3 threads{static_cast<uint32_t>(thread_dims.x),
                   static_cast<uint32_t>(thread_dims.y),
                   static_cast<uint32_t>(thread_dims.z)};
  rt::Status s = metal_stream->rt_stream()->Launch(
      *kernel_, groups, threads, rt_args, static_cast<uint32_t>(shared_bytes));
  if (!s.ok()) return absl::InternalError(s.message());
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace stream_executor
