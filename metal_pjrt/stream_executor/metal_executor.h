// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_STREAM_EXECUTOR_METAL_EXECUTOR_H_
#define METAL_PJRT_STREAM_EXECUTOR_METAL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "xla/stream_executor/blas.h"  // [metal-blas]
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace metal {

// StreamExecutor for one Apple GPU. All memory is unified shared-storage
// MTLBuffer memory, so device, host, collective and unified memory spaces are
// the same thing.
class MetalExecutor : public gpu::GpuExecutor {
 public:
  MetalExecutor(Platform* platform, int device_ordinal);
  ~MetalExecutor() override;

  absl::Status Init() override;

  // Streams and events.
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  void DeallocateStream(Stream* stream) override;
  bool SynchronizeAllActivity() override;

  // Memory.
  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace memory_space) override;
  absl::StatusOr<MemorySpace> GetPointerMemorySpace(const void* ptr) override;
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;
  absl::Status SynchronousMemcpy(DeviceAddressBase* device_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) override;

  // Peer access: there is one GPU and one address space.
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;

  // Kernels and modules. Kernel "binaries" are MSL source; modules are
  // constants containers (see runtime/constants_container.h).
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  absl::StatusOr<DeviceAddressBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;

  // Device description.
  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  // [metal-blas] MPS-backed BLAS/BlasLt (metal_pjrt/blas), created
  // lazily, one per executor.
  blas::BlasSupport* AsBlas() override;

  metal_pjrt::rt::Device* device() const { return device_.get(); }

  using StreamExecutor::Allocate;
  using StreamExecutor::CanEnablePeerAccessTo;
  using StreamExecutor::CreateStream;

 private:
  // Blocks on every live stream; returns the first error.
  absl::Status SynchronizeAllStreams();

  struct LoadedModule {
    int refcount = 0;
    absl::flat_hash_map<std::string, DeviceAddressBase> symbols;
  };

  std::unique_ptr<metal_pjrt::rt::Device> device_;

  mutable absl::Mutex mu_;
  absl::flat_hash_set<Stream*> live_streams_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<const void*, LoadedModule> modules_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<blas::BlasSupport> blas_ ABSL_GUARDED_BY(mu_);  // [metal-blas]
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_STREAM_EXECUTOR_METAL_EXECUTOR_H_
