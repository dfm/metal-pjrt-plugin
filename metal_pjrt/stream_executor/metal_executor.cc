#include "metal_pjrt/stream_executor/metal_executor.h"


#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "metal_pjrt/blas/metal_blas.h"  // [metal-blas]
#include "metal_pjrt/runtime/constants_container.h"
#include "metal_pjrt/compiler/compile_settings.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "metal_pjrt/stream_executor/metal_event.h"
#include "metal_pjrt/stream_executor/metal_kernel.h"
#include "metal_pjrt/stream_executor/metal_stream.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

MetalExecutor::MetalExecutor(Platform* platform, int device_ordinal)
    : gpu::GpuExecutor(platform, device_ordinal) {}

MetalExecutor::~MetalExecutor() {
  // A completion handler still pending after a failure (a GPU reset) would
  // use the device after it is freed: leak the device instead (this happens
  // at process exit).
  if (device_ != nullptr && !device_->WaitForCompletionHandlers()) {
    LOG(ERROR) << "Metal device " << device_ordinal()
               << ": command buffer completion handlers still pending at "
                  "shutdown; leaking the device";
    (void)device_.release();
  }
}

absl::Status MetalExecutor::Init() {
  ABSL_ASSIGN_OR_RETURN(device_, rt::Device::Create(device_ordinal()));
  return absl::OkStatus();
}

// [metal-blas] ---------------------------------------------------------------
// BLAS

blas::BlasSupport* MetalExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  if (blas_ == nullptr && device_ != nullptr) {
    blas_ = std::make_unique<MetalBlas>(device_.get());
  }
  return blas_.get();
}

// ---------------------------------------------------------------------------
// Streams and events

absl::StatusOr<std::unique_ptr<Stream>> MetalExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  ABSL_ASSIGN_OR_RETURN(auto stream, MetalStream::Create(this, priority));
  {
    absl::MutexLock lock(&mu_);
    live_streams_.insert(stream.get());
  }
  return std::unique_ptr<Stream>(std::move(stream));
}

absl::StatusOr<std::unique_ptr<Event>> MetalExecutor::CreateEvent() {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<rt::Event> ev, device_->CreateEvent());
  return std::unique_ptr<Event>(new MetalEvent(std::move(ev)));
}

void MetalExecutor::DeallocateStream(Stream* stream) {
  absl::MutexLock lock(&mu_);
  live_streams_.erase(stream);
}

absl::Status MetalExecutor::SynchronizeAllStreams() {
  std::vector<Stream*> streams;
  {
    absl::MutexLock lock(&mu_);
    streams.assign(live_streams_.begin(), live_streams_.end());
  }
  absl::Status first_error;
  for (Stream* s : streams) {
    absl::Status st = s->BlockHostUntilDone();
    if (!st.ok() && first_error.ok()) first_error = st;
  }
  return first_error;
}

bool MetalExecutor::SynchronizeAllActivity() {
  absl::Status s = SynchronizeAllStreams();
  if (!s.ok()) {
    LOG(ERROR) << "Metal device " << device_ordinal()
               << ": synchronizing all streams failed: " << s;
  }
  return s.ok();
}

// ---------------------------------------------------------------------------
// Memory

DeviceAddressBase MetalExecutor::Allocate(uint64_t size, int64_t memory_space) {
  // Every memory space is unified memory here; the space only matters to XLA's
  // buffer coloring.
  // StreamExecutor contract: failure is a null DeviceAddressBase, so the
  // status is logged here.
  absl::StatusOr<rt::Allocation> a = device_->Allocate(size);
  if (!a.ok()) {
    LOG(ERROR) << "Metal device " << device_ordinal() << ": allocating "
               << size << " bytes (memory space " << memory_space
               << ") failed: " << a.status();
    return DeviceAddressBase();
  }
  return DeviceAddressBase(a->ptr, size);
}

void MetalExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->is_null()) return;
  absl::Status s = device_->Deallocate(mem->opaque());
  if (!s.ok()) {
    LOG(ERROR) << "Metal device " << device_ordinal() << ": deallocating "
               << mem->size() << " bytes at " << mem->opaque()
               << " failed: " << s;
  }
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MetalExecutor::HostMemoryAllocate(uint64_t size) {
  ABSL_ASSIGN_OR_RETURN(rt::Allocation a, device_->Allocate(size));
  return std::make_unique<GenericMemoryAllocation>(
      a.ptr, size, [this](void* ptr, uint64_t size) {
        absl::Status s = device_->Deallocate(ptr);
        if (!s.ok()) {
          LOG(ERROR) << "Metal device " << device_ordinal()
                     << ": freeing host allocation of " << size
                     << " bytes at " << ptr << " failed: " << s;
        }
      });
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MetalExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kDevice:
    case MemorySpace::kUnified:
    case MemorySpace::kCollective:
    case MemorySpace::kHost:
      return std::make_unique<GenericMemoryAllocator>(
          [this](uint64_t size) { return HostMemoryAllocate(size); });
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unsupported memory space %d", static_cast<int>(memory_space)));
  }
}

absl::StatusOr<MemorySpace> MetalExecutor::GetPointerMemorySpace(
    const void* ptr) {
  // Not an error: any pointer outside our allocations is host memory.
  if (device_->Resolve(ptr).ok()) return MemorySpace::kUnified;
  return MemorySpace::kHost;
}

bool MetalExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  // "total" is this process's budget (Device::memory_budget), not the whole
  // GPU working set: on unified memory that is everyone's RAM. (XLA sizes
  // BFC pools, e.g. the collective one, as memory_fraction * total.)
  const rt::Device::MemoryStats m = device_->memory_stats();
  int64_t t = static_cast<int64_t>(m.budget_bytes);
  int64_t used = static_cast<int64_t>(m.live_bytes + m.cached_bytes);
  *total = t;
  *free = t > used ? t - used : 0;
  return true;
}

// Pure virtual in StreamExecutor, called only by multi-device thunks
// (ragged all-to-all); every copy we run goes through a Stream.
absl::Status MetalExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                              const void* host_src,
                                              uint64_t size) {
  return absl::UnimplementedError("Metal: SynchronousMemcpy (use a Stream)");
}

absl::Status MetalExecutor::SynchronousMemcpy(void* host_dst,
                                              const DeviceAddressBase& device_src,
                                              uint64_t size) {
  return absl::UnimplementedError("Metal: SynchronousMemcpy (use a Stream)");
}

absl::Status MetalExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::OkStatus();
}

bool MetalExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return true;
}

// ---------------------------------------------------------------------------
// Kernels and modules

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  if (!spec.has_cuda_cubin_in_memory()) {
    return absl::InvalidArgumentError(
        "Metal kernels must be provided as MSL source in the in-memory "
        "binary field of the KernelLoaderSpec");
  }
  absl::Span<const uint8_t> bytes = spec.cuda_cubin_in_memory()->cubin_bytes;
  std::string msl(reinterpret_cast<const char*>(bytes.data()), bytes.size());

  absl::StatusOr<const rt::Kernel*> rt_kernel =
      device_->GetKernel(msl, spec.kernel_name());
  if (!rt_kernel.ok()) {
    return absl::Status(rt_kernel.status().code(),
                        absl::StrCat("loading kernel ", spec.kernel_name(),
                                     ": ", rt_kernel.status().message()));
  }
  auto kernel = std::make_unique<MetalKernel>(*rt_kernel, spec.arity());
  kernel->set_name(spec.kernel_name());
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  } else {
    const auto& packing_spec =
        std::get<KernelArgsPackingSpec>(spec.kernel_args_packing());
    kernel->set_args_packing(
        [packing_spec](const Kernel& kernel, const KernelArgs& args) {
          const PackableKernelArgs& mem_args =
              dynamic_cast<const PackableKernelArgs&>(args);
          return packing_spec.BuildArguments(mem_args.packed_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return std::unique_ptr<Kernel>(std::move(kernel));
}

absl::StatusOr<ModuleHandle> MetalExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  if (!spec.has_cuda_cubin_in_memory()) {
    return absl::InvalidArgumentError(
        "Metal modules must be constants containers in the in-memory binary "
        "field");
  }
  absl::Span<const uint8_t> bytes = spec.cuda_cubin_in_memory();
  const void* key = bytes.data();
  absl::MutexLock lock(&mu_);
  auto it = modules_.find(key);
  if (it != modules_.end()) {
    ++it->second.refcount;
    return ModuleHandle(key);
  }
  std::vector<rt::ConstantBlob> blobs;
  if (!rt::ParseConstants(bytes.data(), bytes.size(), &blobs)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Metal module bytes (%d bytes) are not a valid constants container",
        bytes.size()));
  }
  LoadedModule module;
  module.refcount = 1;
  for (const rt::ConstantBlob& b : blobs) {
    DeviceAddressBase mem = Allocate(b.data.size(), 0);
    if (mem.is_null() && !b.data.empty()) {
      // Release what this module already allocated.
      for (auto& kv : module.symbols) Deallocate(&kv.second);
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Metal device %d: failed to allocate %d bytes for module constant %s",
          device_ordinal(), b.data.size(), b.name));
    }
    if (!b.data.empty()) std::memcpy(mem.opaque(), b.data.data(), b.data.size());
    module.symbols[b.name] = mem;
  }
  modules_[key] = std::move(module);
  return ModuleHandle(key);
}

bool MetalExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock(&mu_);
  auto it = modules_.find(module_handle.id());
  if (it == modules_.end()) return false;
  if (--it->second.refcount > 0) return true;
  for (auto& kv : it->second.symbols) Deallocate(&kv.second);
  modules_.erase(it);
  return true;
}

absl::StatusOr<DeviceAddressBase> MetalExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  absl::MutexLock lock(&mu_);
  auto it = modules_.find(module_handle.id());
  if (it == modules_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Metal module %p not loaded (looking up symbol %s)",
        module_handle.id(), symbol_name));
  }
  auto sym = it->second.symbols.find(symbol_name);
  if (sym == it->second.symbols.end()) {
    return absl::NotFoundError(
        absl::StrCat("symbol not found in Metal module: ", symbol_name));
  }
  return sym->second;
}

// ---------------------------------------------------------------------------
// Device description

namespace {

// PJRT reports "oneapi <runtime_version>" as the platform version, and JAX's
// persistent compilation cache keys on it. Encode the plugin build and the
// environment variables that change what the compiler emits, so a rebuilt
// plugin or a different setting never loads another's executables:
// {1, fingerprint(build UUID), fingerprint(compile-time settings)}.
// Variables read only at run time (METAL_PJRT_TRACE, ...) are deliberately
// excluded. A new variable that changes compiled code belongs in
// compiler/compile_settings.h.
SemanticVersion PluginVersion() {
  static const SemanticVersion version = [] {
    std::string uuid = metal_pjrt::rt::ImageUuid();
    if (uuid.empty()) LOG(WARNING) << "Metal: plugin image has no LC_UUID";
    // The values the passes use (read once), not the raw environment: so
    // METAL_PJRT_DISABLE_LAPACK=0 and unset share cache entries.
    // (linalg_lowerings.py reads that variable too, but what it lowers is
    // in the HLO, i.e. already in the key.)
    return SemanticVersion(
        1, tsl::Fingerprint32(uuid),
        tsl::Fingerprint32(metal_pjrt::GetCompileSettings().Fingerprint()));
  }();
  return version;
}

}  // namespace

namespace {
std::unique_ptr<DeviceDescription> DescriptionFromInfo(
    const rt::DeviceInfo& info, int device_ordinal) {
  DeviceDescription desc;
  desc.set_name(info.name);
  desc.set_model_str(info.name);
  desc.set_device_vendor("Apple");
  desc.set_platform_version("Metal");
  desc.set_driver_version(SemanticVersion{0, 0, 0});
  desc.set_runtime_version(PluginVersion());
  desc.set_compile_time_toolkit_version(SemanticVersion{0, 0, 0});
  desc.set_dnn_version(SemanticVersion{0, 0, 0});
  desc.set_pci_bus_id(absl::StrCat("metal:", device_ordinal));
  desc.set_numa_node(0);

  // Thread geometry. Apple GPUs execute 32-wide SIMD groups; a threadgroup
  // holds up to 1024 threads.
  desc.set_threads_per_warp(info.simd_width);
  desc.set_threads_per_block_limit(info.max_threads_per_threadgroup);
  desc.set_threads_per_core_limit(info.max_threads_per_threadgroup);
  desc.set_thread_dim_limit(ThreadDim(info.max_threads_per_threadgroup,
                                      info.max_threads_per_threadgroup,
                                      info.max_threads_per_threadgroup));
  const int64_t kMaxGrid = (1ll << 31) - 1;
  desc.set_block_dim_limit(BlockDim(kMaxGrid, kMaxGrid, kMaxGrid));
  desc.set_max_blocks_per_multiprocessor(32);

  // Memory. Threadgroup ("shared") memory is 32 KB on all Apple GPUs so far.
  desc.set_shared_memory_per_block(info.threadgroup_memory_length);
  desc.set_shared_memory_per_block_optin(info.threadgroup_memory_length);
  desc.set_shared_memory_per_core(info.threadgroup_memory_length);
  desc.set_reserved_shared_memory_per_block(0);
  desc.set_registers_per_core_limit(65536);
  desc.set_registers_per_block_limit(65536);
  desc.set_device_address_bits(64);
  desc.set_device_memory_size(info.recommended_working_set);
  desc.set_l2_cache_size(8 << 20);
  // Metal exposes neither bandwidth nor clocks; these only feed cost models.
  desc.set_memory_bandwidth(100e9);
  desc.set_clock_rate_ghz(1.4f);
  desc.set_core_count(10);
  desc.set_fpus_per_core(128);
  desc.set_ecc_enabled(false);

  // v1 reports an OneAPI compute capability so that GpuCompiler and the MLIR
  // emitters take their SPIR-V (vendor-neutral) branches. The vendor string
  // above is what identifies the device as Metal to our own code.
  desc.set_oneapi_compute_capability(static_cast<uint32_t>(info.gpu_family));

  return std::make_unique<DeviceDescription>(std::move(desc));
}
}  // namespace

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription() const {
  return DescriptionFromInfo(device_->info(), device_ordinal());
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription(int device_ordinal) {
  ABSL_ASSIGN_OR_RETURN(rt::DeviceInfo info,
                        rt::Device::QueryInfo(device_ordinal));
  return DescriptionFromInfo(info, device_ordinal);
}

}  // namespace metal
}  // namespace stream_executor
