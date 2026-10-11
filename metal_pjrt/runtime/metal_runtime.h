// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Thin C++ runtime over Metal (via metal-cpp), independent of XLA (it depends
// only on absl) so it can be unit-tested standalone. The StreamExecutor adapter translates this API into
// XLA's abstractions.
//
// Model:
//   Device   one MTL::Device + allocator bookkeeping + the kernel cache, and
//            the device's one MTL::CommandQueue: GPU work of every stream is
//            encoded into its open command buffer, committed at sync points
//            or when a budget is reached. Each committed buffer ends by
//            signaling the device timeline (an MTL::SharedEvent) with its
//            sequence number, so a timeline value names all GPU work up to
//            that buffer.
//   Stream   a handle on the queue: the timeline value of its last GPU op,
//            the waits its next op must respect, and a worker thread for its
//            host tasks (each signals the stream's host fence when done).
//   Event    a WorkPoint: a timeline value plus host-task values.
//
// No command buffer ever waits on the GPU: GPU work that depends on other
// GPU work is behind it in the one queue, and GPU work that depends on a
// host task is encoded only once the task has run (the encoding thread
// waits on the host, without holding the queue lock). Host tasks wait on
// the host for the timeline, and the work they wait for is committed
// before they are enqueued.
//
// Device memory is plain MTL::Buffer with shared storage (unified memory), so
// a "device pointer" is the buffer's contents() address. XLA hands us raw
// pointers, so Device keeps an address-ordered map to resolve a pointer back to
// (buffer, offset).
//
// Errors are absl::Status with these codes:
//   InternalError           a Metal API call failed (message carries the
//                           NSError description, domain and code)
//   ResourceExhaustedError  an allocation failed (requested size, limits)
//   InvalidArgumentError    caller mistake (unknown pointer, bad size/args)
//   UnimplementedError      known gap (e.g. more than 31 buffer arguments
//                           for a kernel without an argument buffer)
//   FailedPreconditionError a quarantined kernel, or a host task waited for
//                           itself
//
// GPU failures are sticky for the device, as a CUDA context error is for the
// context: the first failed command buffer (whatever the error: watchdog
// timeout, page fault, out of memory, ...) or host task without an error
// callback sets Device::error(), and from then on every Synchronize,
// Event::WaitOnHost/Poll, host task and new launch gets that error. The
// output of failed work is therefore never consumed as valid, and there is
// no recovery short of restarting the process (PJRT never synchronizes its
// compute stream, so under JAX a failure was effectively sticky anyway).
// Host waiters do not wait for completion handlers to learn about a
// failure: after seeing a signal they check the status of the command
// buffers still in flight (Device::CheckInFlight).
#ifndef METAL_PJRT_RUNTIME_METAL_RUNTIME_H_
#define METAL_PJRT_RUNTIME_METAL_RUNTIME_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt/codegen/msl_kernel.h"

namespace MTL {
class Device;
class Buffer;
class Library;
class ComputePipelineState;
class CommandQueue;
class CommandBuffer;
class ComputeCommandEncoder;
class SharedEvent;
}  // namespace MTL

namespace metal_pjrt {
namespace rt {

struct DeviceInfo {
  std::string name;
  uint64_t max_buffer_length = 0;          // largest single MTL::Buffer
  uint64_t recommended_working_set = 0;    // bytes
  uint32_t max_threads_per_threadgroup = 1024;
  uint32_t threadgroup_memory_length = 32768;
  uint32_t simd_width = 32;
  int gpu_family = 0;                      // highest MTLGPUFamilyAppleN supported
};

// A device allocation. `ptr` is the host-visible/device-visible address.
struct Allocation {
  void* ptr = nullptr;
  uint64_t size = 0;
};

// Resolved location of a raw device pointer.
struct BufferRef {
  MTL::Buffer* buffer = nullptr;
  uint64_t offset = 0;
};

struct Dim3 {
  uint32_t x = 1, y = 1, z = 1;
};

// One kernel argument. Exactly one of {buffer, bytes} is used.
struct KernelArg {
  static KernelArg Buffer(const void* device_ptr) {
    KernelArg a;
    a.is_buffer = true;
    a.device_ptr = device_ptr;
    return a;
  }
  static KernelArg Bytes(const void* data, size_t len) {
    KernelArg a;
    a.is_buffer = false;
    a.bytes.assign(static_cast<const uint8_t*>(data),
                   static_cast<const uint8_t*>(data) + len);
    return a;
  }
  bool is_buffer = false;
  const void* device_ptr = nullptr;
  // Inline so building a launch's arguments does not touch the heap.
  absl::InlinedVector<uint8_t, 48> bytes;
};

// MSL kernels whose source contains this marker take their buffer arguments
// through an argument buffer: [[buffer(0)]] is `constant ulong*`, one 64-bit
// GPU address (MTLBuffer.gpuAddress + offset) per argument. Used for kernels
// with more buffer arguments than Metal's argument table holds.
using codegen::kArgumentBufferMarker;

// An MSL source starting with this line stands for the same source with the
// line replaced by kernels/msl_prelude.metal; GetKernel/AcquireKernel compile
// that (the cache keeps the short form). The MSL emitter's kernels start with
// it.
using codegen::kMslPreludeLine;

// True when kernel `kernel_name` in `msl_source` uses the argument-buffer
// convention (the marker line immediately precedes `kernel void <name>(`).
bool UsesArgumentBuffer(const std::string& msl_source,
                        const std::string& kernel_name);

// N from a `[[max_total_threads_per_threadgroup(N)]]` attribute written just
// before `kernel void <name>(` (optionally with the argument-buffer marker in
// between), or 0 if there is none.
uint32_t DeclaredMaxThreadsPerThreadgroup(const std::string& msl_source,
                                          const std::string& kernel_name);

// What identifies a kernel across processes: its function name and a key
// made of the hash of its MSL source plus the function name. Shared by the
// Kernel and by the streams' per-command-buffer records (so recording a
// launch costs a reference count, not a string copy).
struct KernelIdentity {
  std::string name;
  std::string key;
  // Device::BuiltinKernel's fill/copy kernels: in nearly every command
  // buffer, so never blamed for a reset.
  bool builtin = false;
};

// The LC_UUID of the Mach-O image containing the runtime (the plugin dylib,
// or a test binary): 16 raw bytes, or empty. The linker derives it from the
// image's contents, so every rebuild that changes the code gets a new one.
std::string ImageUuid();

// The reason of the last allocation Device::Allocate refused (any device),
// its requested size and the executable it was for (NameAllocationRefusal;
// "" when not known), or "" when none was refused. XLA's allocator adapter
// turns every refusal into a generic "Out of memory while trying to allocate
// N" error; the plugin's PJRT_Error_Message appends this reason to it
// (pjrt/metal_pjrt_api.cc).
std::string LastAllocationRefusal(uint64_t* size, std::string* executable);

// The number of the last refusal on the calling thread (0: none). Refusals
// are numbered in order across threads.
uint64_t LastAllocationRefusalOnThisThread();

// Records that refusal number `refusal` was for `executable`, if it is still
// the last one.
void NameAllocationRefusal(uint64_t refusal, std::string executable);

// A compute pipeline, owned by the Device's kernel cache (Device::GetKernel,
// Device::AcquireKernel).
class Kernel {
 public:
  Kernel(MTL::ComputePipelineState* pso,
         std::shared_ptr<const KernelIdentity> identity,
         bool uses_argument_buffer, uint32_t declared_max_threads)
      : pso_(pso),
        identity_(std::move(identity)),
        uses_argument_buffer_(uses_argument_buffer),
        declared_max_threads_(declared_max_threads) {}
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  MTL::ComputePipelineState* pso() const { return pso_; }
  const std::string& name() const { return identity_->name; }
  const std::string& key() const { return identity_->key; }
  const std::shared_ptr<const KernelIdentity>& identity() const {
    return identity_;
  }
  // See kArgumentBufferMarker (UsesArgumentBuffer on the source). Launch
  // then packs all (buffer) arguments into one setBytes payload of GPU
  // addresses and marks the buffers resident.
  bool uses_argument_buffer() const { return uses_argument_buffer_; }
  // Includes the source's declared [[max_total_threads_per_threadgroup]]
  // (DeclaredMaxThreadsPerThreadgroup): release Metal does not check
  // dispatches against it (a larger threadgroup is undefined behaviour on
  // the GPU), so launches above it are refused before encoding.
  uint32_t max_total_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;

 private:
  friend class Device;
  MTL::ComputePipelineState* pso_;
  std::shared_ptr<const KernelIdentity> identity_;
  bool uses_argument_buffer_;
  uint32_t declared_max_threads_;  // 0: none declared
  // Kernel cache bookkeeping, guarded by the Device's kernel_mu_: the
  // source (its cache key) and function + constants (the key within it),
  // the AcquireKernel references, and whether GetKernel returned it (then
  // it stays until ~Device).
  absl::string_view source_;
  std::string cache_key_;
  int refs_ = 0;
  bool pinned_ = false;
};

// The value of an MSL function constant (`constant T name
// [[function_constant(index)]]`), fixed when the pipeline is created.
struct FunctionConstant {
  enum class Type { kBool, kInt };  // MSL bool, int
  static FunctionConstant Bool(uint32_t index, bool v) {
    return {index, Type::kBool, v ? 1 : 0};
  }
  static FunctionConstant Int(uint32_t index, int32_t v) {
    return {index, Type::kInt, v};
  }
  uint32_t index = 0;
  Type type = Type::kInt;
  int32_t value = 0;
};

class Event;
class Stream;

// A position in a device's work: GPU work up to device timeline value `t`
// (0: none) and host tasks up to `value` on their stream's host fence.
struct WorkPoint {
  uint64_t t = 0;
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> hosts;
  // Both this and `other`.
  void Merge(const WorkPoint& other);
};

class Device {
 public:
  static absl::StatusOr<std::unique_ptr<Device>> Create(int ordinal);
  // The info a Device for `ordinal` would have, without creating one.
  static absl::StatusOr<DeviceInfo> QueryInfo(int ordinal);
  static int VisibleDeviceCount();
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Waits (up to kHandlerWait) for the completion handlers of committed
  // command buffers, which use this Device; false if some are still pending
  // (after a failure, e.g. a reset, they may never run). ~Device waits too,
  // but an owner that gets false should leak the Device rather than destroy
  // it: a handler running later would use freed memory.
  bool WaitForCompletionHandlers();

  const DeviceInfo& info() const { return info_; }

  // Memory this process may hold (live + cached): half of physical RAM,
  // capped by the GPU's recommended working set, times
  // METAL_PJRT_MEMORY_FRACTION (default 1).
  uint64_t memory_budget() const { return memory_budget_; }

  // The first GPU (or unhandled host-task) failure; sticky for the
  // process (see the file comment). Continuing to submit work to a GPU that
  // is being reset makes recovery less likely, so new work is refused.
  absl::Status error() const;
  void SetError(const absl::Status& why);
  int ordinal() const { return ordinal_; }
  MTL::Device* mtl() const { return device_; }

  // Memory. Allocations are shared-storage MTL::Buffers; freeing an address
  // that was not returned by Allocate is an error.
  //
  // Freed buffers are cached by size and handed out again immediately (as
  // XLA's BFC pool does: reuse is ordered by the compute stream), because a
  // fresh MTL::Buffer costs ~60 us/MB of page faults on first touch. A miss
  // allocates a new buffer if live + cached + size stays within
  // memory_budget() (evicting least recently freed buffers to make room) and
  // CheckSystemMemory passes; otherwise RESOURCE_EXHAUSTED. Cached buffers
  // are released when unused for kCacheIdleRelease, and all at once on a
  // system memory-pressure warning (while the level is not normal, frees are
  // released as soon as allowed), so an idle process gives its memory back.
  // A cached buffer is only released (evicted, trimmed or dropped) once all
  // work that existed when it was freed has finished (see BeginWork): host
  // tasks hold raw pointers, and a command buffer may use it through an
  // argument buffer. Until then it stays cached (and may briefly push live +
  // cached over the budget); reuse needs no wait, being stream-ordered.
  //
  // kHostWrite: for a buffer the host writes into right away, off any
  // stream (module constants, FFT tables). It takes only a cached buffer
  // whose release ticket has ended (no work that existed when it was freed
  // is left, see FindCachedBuffer), else a fresh one: a reused buffer that
  // queued GPU work still reads would see the host's bytes.
  enum class Use { kGpu, kHostWrite };
  absl::StatusOr<Allocation> Allocate(uint64_t size, Use use = Use::kGpu);
  // The one system-level limit, for Allocate and buffers made outside it:
  // an allocation of 1 MB or more is refused (RESOURCE_EXHAUSTED) while the
  // system is at critical memory pressure after dropping the cache (see
  // CriticalAfterReleasingCache). Below that macOS pages as for any process
  // (as PyTorch MPS allows); the first allocation at warning or worse logs
  // a warning.
  absl::Status CheckSystemMemory(uint64_t size);
  // Frees an allocation; `size`, when given, must be the size it was
  // allocated with. A pointer that is not the start of a live allocation
  // (unknown, interior, already freed) or a size mismatch is a plugin bug:
  // INTERNAL, logged with a backtrace, and nothing is cached or released.
  // METAL_PJRT_DEBUG_FREE_QUARANTINE=1 (debugging): freed buffers are not
  // reused until kQuarantineFrees later frees and kQuarantineTime have
  // passed, and a pointer into one is reported as such, with the
  // backtrace of its free.
  absl::Status Deallocate(void* ptr, std::optional<uint64_t> size = {});
  static constexpr int kQuarantineFrees = 64;
  static constexpr std::chrono::milliseconds kQuarantineTime{100};
  // Host memory the GPU can also reach (XLA's host memory space: its host
  // staging pool and pinned_host arrays, which it copies to and from with
  // device copies): anonymous pages wrapped without copying in an
  // MTL::Buffer. Not device memory: outside the budget, the cache and
  // Allocate's table (Resolve finds it all the same). The pages are unmapped
  // when the buffer goes, i.e. after DeallocateHost and once no command
  // buffer holds it. Beyond maxBufferLength the pages are host-only.
  // RESOURCE_EXHAUSTED only if the pages cannot be mapped.
  absl::StatusOr<void*> AllocateHost(uint64_t size);
  absl::Status DeallocateHost(void* ptr);
  // Whether `ptr` lies in AllocateHost's memory.
  bool IsHostMemory(const void* ptr) const;
  // Resolve a raw pointer (possibly interior) to its buffer and offset.
  absl::StatusOr<BufferRef> Resolve(const void* ptr) const;
  // The live allocations nearest to `ptr` (below and above), for errors.
  std::string DescribeNearestAllocations(const void* ptr) const;
  // Live bytes (allocated, not freed; buffer lengths).
  uint64_t allocated_bytes() const;
  static constexpr std::chrono::seconds kCacheIdleRelease{2};
  struct MemoryStats {
    uint64_t live_bytes = 0;
    uint64_t peak_live_bytes = 0;
    uint64_t num_allocs = 0;  // successful allocations, ever
    uint64_t cached_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t released = 0;  // buffers released (not recycled), ever
    int pressure = 0;  // 0 normal, 1 warning, 2 critical
  };
  MemoryStats memory_stats() const;
  // What the DISPATCH_SOURCE_TYPE_MEMORYPRESSURE handler calls (level as
  // above); tests call it through metal_pjrt_testing_memory_pressure().
  void OnMemoryPressure(int level);
  // Releases cached buffers freed at least `min_idle` ago (all with 0) whose
  // work has finished.
  void TrimCache(std::chrono::steady_clock::duration min_idle);
  // Work tickets: every command buffer holds one from creation to
  // completion, every host task from enqueue to its end.
  uint64_t BeginWork();
  void EndWork(uint64_t ticket);

  // The kernel `function` of the Metal Shading Language source with
  // `constants` bound. The one kernel cache: compiled (fast math off, MSL
  // 4.0) on first use and kept, keyed by the full source, the function name
  // and the constants, so the returned kernel lives as long as the device. A
  // quarantined kernel (see RecordReset) is refused, also when cached.
  absl::StatusOr<const Kernel*> GetKernel(
      absl::string_view msl_source, absl::string_view function,
      absl::Span<const FunctionConstant> constants = {});
  // Like GetKernel, but the kernel is counted, not kept: it stays cached
  // until every AcquireKernel is matched by a ReleaseKernel (unless GetKernel
  // returned it too), and the source's library and cached text go with its
  // last kernel. For kernels owned by compiled executables (the
  // StreamExecutor's LoadKernel), so dropping an executable frees them.
  // Command buffers retain their pipelines, so work in flight is not
  // affected; KernelIdentity (reset attribution, quarantine) outlives the
  // kernel.
  absl::StatusOr<const Kernel*> AcquireKernel(absl::string_view msl_source,
                                              absl::string_view function);
  void ReleaseKernel(const Kernel* kernel);
  struct KernelCacheStats {
    uint64_t libraries = 0;     // MSL sources
    uint64_t kernels = 0;       // pipelines
    uint64_t source_bytes = 0;  // MSL text kept as cache keys
  };
  KernelCacheStats kernel_cache_stats();

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream();
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent();

  // Built-in kernels (compiled on first use): fills at 32-bit and byte
  // granularity, copies in 16-byte vectors and bytes. Small copies and fills
  // run as compute kernels so a stream of kernels and copies stays in one
  // compute encoder (creating a blit encoder per copy cost ~10 us and an
  // encoder switch on both sides); large ones use the blit engine.
  enum class Builtin { kFill32, kFill8, kCopy16, kCopy8 };
  absl::StatusOr<const Kernel*> BuiltinKernel(Builtin kind);

  // GPU reset log and kernel quarantine. A watchdog timeout resets the GPU
  // for every process, and after a few resets the driver leaves the GPU slow
  // until a reboot, so the same bug must not be allowed to reset it again
  // and again. Every reset observed by this process is appended to
  // <state dir>/gpu_resets.jsonl with the time, the plugin build (ImageUuid,
  // diagnostics only) and the kernels that were in the command buffer that
  // timed out (built-in kernels are listed separately and never blamed). Kernels seen in
  // `quarantine_strikes()` or more resets since boot are refused by
  // GetKernel until a reboot or until the log is cleared
  // (scripts/gpu_health.py --clear). A kernel whose source changes gets a
  // new key; a fix elsewhere (runtime, launch dimensions) needs --clear.
  // The state directory is METAL_PJRT_STATE_DIR or ~/.cache/metal-pjrt;
  // METAL_PJRT_QUARANTINE_STRIKES sets the threshold (default 0: off).
  void RecordReset(absl::string_view cause,
                   absl::Span<const std::shared_ptr<const KernelIdentity>>
                       kernels);
  const std::string& state_dir() const { return state_dir_; }
  int quarantine_strikes() const { return quarantine_strikes_; }
  // Resets recorded since the machine booted (by any process; this one's
  // RecordReset counts too).
  int resets_since_boot() const { return resets_since_boot_; }

  // GPU ops whose encoding waited on the host for a host task.
  uint64_t encode_host_waits() const { return encode_host_waits_.load(); }

  // Waits on the host until `p` is done, committing the open command
  // buffer first if `p` includes work in it. A failure on the GPU ends the
  // wait with the device error; host tasks always end (after a failure they
  // skip or report, see Stream::HostCallback), so their values are waited
  // for even then. FailedPrecondition when a host task would wait for
  // itself or later work of its stream.
  absl::Status WaitForPoint(const WorkPoint& p);
  // Non-blocking: whether `p` is done (committed and signaled).
  bool PointDone(const WorkPoint& p) const;

 private:
  Device() = default;
  int ordinal_ = 0;
  MTL::Device* device_ = nullptr;
  DeviceInfo info_;

  mutable std::mutex mu_;
  struct LiveAllocation {
    MTL::Buffer* buffer;
    uint64_t length;      // the buffer's (rounded) length
    uint64_t requested;   // the size asked for
    uint64_t generation;  // allocation sequence number (diagnostics)
  };
  // Keyed by start address.
  std::map<uintptr_t, LiveAllocation> allocations_;
  uint64_t generation_ = 0;
  // The last frees (address, size, generation), for telling a double free
  // from an unknown pointer.
  struct RecentFree {
    uintptr_t addr;
    uint64_t requested;
    uint64_t generation;
  };
  std::deque<RecentFree> recent_frees_;
  static constexpr size_t kRecentFrees = 1024;
  // Debug quarantine (see Deallocate), oldest first.
  struct QuarantinedBuffer {
    LiveAllocation a;
    uintptr_t addr;
    std::chrono::steady_clock::time_point freed;
    uint64_t free_seq;
    uint64_t ticket;
    std::vector<void*> free_stack;
  };
  bool quarantine_frees_ = false;  // set by Create
  std::deque<QuarantinedBuffer> quarantine_;
  uint64_t free_seq_ = 0;
  // Moves quarantined buffers that have served their time into the cache
  // (`idle`, from TrimCache: kQuarantineTime alone).
  void DrainQuarantineLocked(bool idle = false);
  // A pointer bug: logs `msg` with a backtrace and what is known about
  // `ptr`, returns INTERNAL.
  absl::Status PointerBug(const void* ptr, const std::string& msg);
  // AllocateHost's buffers, keyed as allocations_ is (length = mapped bytes).
  std::map<uintptr_t, std::pair<MTL::Buffer*, uint64_t>> host_allocations_;
  uint64_t allocated_bytes_ = 0;
  uint64_t peak_allocated_bytes_ = 0;
  uint64_t num_allocs_ = 0;
  uint64_t memory_budget_ = 0;
  // The free-buffer cache (see Allocate), guarded by mu_: least recently
  // freed first, and indexed by length.
  struct CachedBuffer {
    MTL::Buffer* buffer;
    uint64_t size;
    std::chrono::steady_clock::time_point freed;
    uint64_t ticket;  // last ticket issued when freed
  };
  std::list<CachedBuffer> cache_;
  std::multimap<uint64_t, std::list<CachedBuffer>::iterator> cache_by_size_;
  uint64_t cached_bytes_ = 0;
  uint64_t cache_hits_ = 0;
  uint64_t cache_misses_ = 0;
  uint64_t released_ = 0;
  int pressure_ = 0;
  // Removes `it` from the cache and appends its buffer to `out`.
  void EvictLocked(std::list<CachedBuffer>::iterator it,
                   std::vector<MTL::Buffer*>* out);
  // Releases buffers taken out of the cache (outside mu_).
  void ReleaseBuffers(const std::vector<MTL::Buffer*>& buffers);
  // Outstanding work tickets; lock order mu_ -> tickets_mu_.
  std::mutex tickets_mu_;
  std::set<uint64_t> outstanding_;
  uint64_t last_ticket_ = 0;
  // Every ticket below this has ended.
  uint64_t EndedBelow();
  std::condition_variable tickets_cv_;  // notified by EndWork
  // At critical pressure: drops the cache and reads the level again.
  // Buffers freed while work was in flight (e.g. the previous step's) stay
  // cached until that work ends, so if some remain, waits (bounded by
  // kRefusalWait, not after a device error) for the work outstanding now,
  // drops the cache and reads once more (no wait under the queue lock, i.e.
  // inside EncodeExternal). Returns whether still critical.
  bool CriticalAfterReleasingCache();
  static constexpr std::chrono::seconds kRefusalWait{1};
  // libdispatch sources on memory_queue_: the memory-pressure source and a
  // timer running TrimCache(kCacheIdleRelease), resumed only while the cache
  // is not empty (trim_armed_, guarded by mu_).
  void* memory_queue_ = nullptr;     // dispatch_queue_t
  void* pressure_source_ = nullptr;  // dispatch_source_t
  void* trim_timer_ = nullptr;       // dispatch_source_t
  bool trim_armed_ = false;
  bool stopping_ = false;  // ~Device has started; guarded by mu_
  // How long ~Device waits for completion handlers still pending.
  static constexpr std::chrono::seconds kHandlerWait{5};
  void StartMemorySources();
  absl::Status error_ = absl::OkStatus();
  // The kernel cache: per MSL source, its library, its identity hash and
  // its kernels by function name + constants. Guarded by kernel_mu_, a leaf
  // lock (compilation happens outside it). An entry goes when its last
  // kernel is released and no GetKernelImpl is compiling from it (pending).
  struct CachedLibrary {
    MTL::Library* library = nullptr;
    std::string hash;  // HashString(source): the kernels' KernelIdentity key
    absl::flat_hash_map<std::string, std::unique_ptr<Kernel>> kernels;
    int pending = 0;
  };
  // `pin`: GetKernel (kept until ~Device), else AcquireKernel (counted).
  absl::StatusOr<const Kernel*> GetKernelImpl(
      absl::string_view msl_source, absl::string_view function,
      absl::Span<const FunctionConstant> constants, bool builtin, bool pin);
  // Takes a pin or a reference on `k`; kernel_mu_ held.
  static void HoldLocked(Kernel* k, bool pin);
  // Drops the entry of `source` if it has no kernels and nothing pending;
  // kernel_mu_ held. Returns its library to release (outside the lock), or
  // null.
  MTL::Library* MaybeEraseLocked(absl::string_view source);
  // FailedPrecondition when `id` is quarantined. Takes mu_.
  absl::Status CheckQuarantine(const KernelIdentity& id);
  std::mutex kernel_mu_;
  absl::node_hash_map<std::string, CachedLibrary> kernel_cache_;
  std::mutex builtin_mu_;
  const Kernel* builtin_kernels_[4] = {};
  // Reset log state (see RecordReset); strikes_ is guarded by mu_.
  void LoadResetLog();
  std::string state_dir_;
  int quarantine_strikes_ = 0;  // off unless METAL_PJRT_QUARANTINE_STRIKES
  int resets_since_boot_ = 0;
  std::unordered_map<std::string, int> strikes_;  // kernel key -> resets
  // Whether any kernel could be quarantined (strikes_ non-empty and the
  // quarantine on), so cache hits skip mu_ otherwise.
  std::atomic<bool> may_quarantine_{false};
  // Set by the first failed buffer that records a reset: one record per
  // incident (a reset fails every buffer in flight; the device error is
  // sticky, so there is no second incident in this process).
  std::atomic<bool> reset_recorded_{false};
  std::string build_;  // hex ImageUuid, recorded with each reset
  friend class Stream;
  friend class Event;
  // Committed command buffers whose completion handler has not run yet
  // (retained, with the timeline value they end by signaling; `injected` is
  // FailNextCommandBufferForTesting's error, which counts as the buffer's
  // status). Completion handlers remove theirs; they never block.
  struct InFlight {
    MTL::CommandBuffer* cb;
    uint64_t value;
    absl::Status injected;
  };
  std::mutex in_flight_mu_;
  std::vector<InFlight> in_flight_;
  void AddInFlight(MTL::CommandBuffer* cb, uint64_t value,
                   absl::Status injected);
  void RemoveInFlight(MTL::CommandBuffer* cb);
  // Called by host waiters after they saw a signal: sets error() if any
  // in-flight command buffer has failed. A buffer whose timeline value is
  // signaled has run to its end, so with `wait` its final status is waited
  // for (Metal sets it before running handlers); without, returns false if
  // some such status is not final yet. This covers every buffer the waiter
  // depends on without waiting for handlers: one that did not run to its
  // signal is force-signaled by its handler after the error is recorded.
  bool CheckInFlight(bool wait);
  std::atomic<uint64_t> encode_host_waits_{0};

  // The queue. Guarded by q_mu_; lock order Stream::mu_ -> q_mu_ -> the
  // other device locks. Nothing waits while holding q_mu_.
  std::mutex q_mu_;
  MTL::CommandQueue* queue_ = nullptr;
  MTL::SharedEvent* timeline_ = nullptr;
  // The last committed buffer's timeline value; the open buffer's is one
  // more. Written under q_mu_, read anywhere.
  std::atomic<uint64_t> committed_{0};
  // After a device error the open buffer stays open (its ticket
  // outstanding, so buffers freed meanwhile stay cached) until a wait
  // commits it.
  MTL::CommandBuffer* cmd_ = nullptr;
  uint64_t cmd_ticket_ = 0;  // BeginWork for cmd_
  std::atomic<uint64_t> open_ticket_{0};  // cmd_ticket_ while cmd_ is open
  MTL::ComputeCommandEncoder* enc_ = nullptr;
  int ops_in_cmd_ = 0;
  uint64_t threads_in_cmd_ = 0;
  uint64_t flops_in_cmd_ = 0;
  // Kernels encoded into the open buffer (failure diagnostics, reset log).
  std::vector<std::shared_ptr<const KernelIdentity>> pending_kernels_;
  absl::Status inject_error_;  // Stream::FailNextCommandBufferForTesting
  // Streams' host fences, kept while the device lives: work points
  // (events, other streams' waits) may name them after their stream is gone.
  std::vector<MTL::SharedEvent*> host_fences_;
  // Queue operations; caller holds q_mu_.
  absl::Status EnsureCommandBufferLocked();
  absl::Status EnsureComputeEncoderLocked();
  void EndEncoderLocked();
  // Commits the open buffer. After a device error nothing is committed: the
  // buffer is dropped and its timeline value force-signaled, as a failed
  // buffer's handler does.
  absl::Status CommitLocked();
  // Before encoding an op of `flops`: commits first if it would take the
  // open buffer over kMaxFlopsPerCommandBuffer.
  absl::Status MakeRoomLocked(uint64_t flops);
  // Accounts one encoded op and commits per the batching policy (see
  // Stream::kMaxOpsPerCommandBuffer).
  absl::Status FinishOpLocked(uint64_t work, uint64_t flops);
  absl::Status EncodeBuiltinLocked(Builtin kind,
                                   absl::Span<const BufferRef> buffers,
                                   const void* bytes, size_t bytes_len,
                                   uint64_t n, uint64_t* work);
  absl::Status EncodeCopyLocked(BufferRef dst, BufferRef src, uint64_t size,
                                uint64_t* work);
  absl::Status EncodeFillLocked(BufferRef dst, uint32_t pattern,
                                int pattern_bytes, uint64_t size,
                                uint64_t* work);
  // Commits the open buffer if timeline value `t` is its.
  absl::Status CommitThrough(uint64_t t);
};

// An event is a recorded WorkPoint: everything enqueued on the stream
// before RecordEvent. Waiting for it on another stream makes that stream's
// later work wait (see Stream::WaitForEvent).
class Event {
 public:
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  // True once the recorded work is done (also on failure).
  bool IsComplete() const;
  // Non-blocking: false while pending, true once complete, or the device's
  // sticky error.
  absl::StatusOr<bool> Poll();
  // Block the host until complete (see Device::WaitForPoint).
  absl::Status WaitOnHost();

 private:
  friend class Device;
  friend class Stream;
  explicit Event(Device* device) : device_(device) {}
  WorkPoint point() const;
  Device* device_;
  mutable std::mutex mu_;
  WorkPoint point_;  // empty until recorded
};

class Stream {
 public:
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  // Kernel launch. threadgroups = grid in threadgroup units, threads = per
  // threadgroup. Buffer args are resolved to (buffer, offset) here. `flops`
  // is the launch's cost for the kMaxFlopsPerCommandBuffer budget (GEMMs;
  // blas::GemmWork), 0 for kernels whose cost the thread count describes.
  absl::Status Launch(const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
                      absl::Span<const KernelArg> args,
                      uint32_t threadgroup_memory_bytes = 0,
                      uint64_t flops = 0);

  // Metal's per-stage buffer argument table has 31 slots. Kernels using an
  // argument buffer (Kernel::uses_argument_buffer) take up to
  // kMaxArgumentBufferArgs buffer arguments (4 KB of setBytes payload).
  static constexpr size_t kMaxBufferArgs = 31;
  static constexpr size_t kMaxArgumentBufferArgs = 4096 / sizeof(uint64_t);

  // Encode work produced outside this runtime (e.g. Metal Performance
  // Shaders) into the device's open command buffer, in order with all other
  // work. Ends the current compute encoder, then calls `encode` with the
  // raw MTL::CommandBuffer* (as void*, so Objective-C++ callers can bridge it
  // to id<MTLCommandBuffer>). `encode` must create, use and end its own
  // encoders, also when it returns an error (the next op opens its encoder
  // on the same buffer, and two open encoders are invalid; what it did
  // encode stays in the buffer), and must not commit the buffer or call
  // back into the runtime (the queue lock is held). Counts as one op toward
  // kMaxOpsPerCommandBuffer and `flops` toward kMaxFlopsPerCommandBuffer.
  absl::Status EncodeExternal(
      std::function<absl::Status(void* mtl_command_buffer)> encode,
      uint64_t flops);

  // Device-to-device copy and fill. Copies and fills up to
  // kComputeCopyMaxBytes, and fills whose pattern is not one repeated byte,
  // run as the device's built-in compute kernels; larger ones use a blit
  // encoder.
  absl::Status MemcpyDeviceToDevice(void* dst, const void* src, uint64_t size);
  absl::Status Memset8(void* dst, uint8_t value, uint64_t size);
  absl::Status Memset32(void* dst, uint32_t value, uint64_t size);

  // Host transfers are ordered on the stream. With unified memory they are
  // plain memcpys: done right away on the calling thread when everything
  // the stream has enqueued or waits for is done; otherwise as a host task
  // (so `src` must stay unchanged until the stream gets there, as XLA
  // guarantees; pjrt/metal_pjrt_api.cc snapshots device_put's numpy data,
  // or waits for the copy of a large array). After a device failure the
  // task skips the copy (XLA may already be freeing its buffers).
  absl::Status MemcpyHostToDevice(void* dst, const void* src, uint64_t size);
  absl::Status MemcpyDeviceToHost(void* dst, const void* src, uint64_t size);
  // The device side of a host transfer must be `size` bytes inside a live
  // allocation: anything else means the allocation table lost a buffer XLA
  // still holds (a use after free), so the copy is refused with an
  // InternalError, and logged, instead of writing through the pointer.
  absl::Status CheckHostTransfer(const void* device_ptr, uint64_t size,
                                 absl::string_view what);

  // Run `fn` on the stream's worker thread once everything enqueued on the
  // stream (or that it waits for) is done; later work on the stream waits
  // for it (GPU work is encoded only after it ran). A task must not wait for
  // its own stream: every method of that stream called from the task (or
  // its `on_error`), and host waits on the task or later work of its stream,
  // return FailedPreconditionError instead of deadlocking. Nor may a task
  // call into any other stream: an encoding thread may be host-waiting for
  // the task, a cycle (no XLA path does this; it is not checked).
  //
  // After a failure (the device's sticky error), a task with `on_error` does
  // not run and `on_error` gets the error; one without it still runs (XLA's
  // plain callbacks free memory or complete transfers). Either way it is
  // enqueued and HostCallback returns OK. If `fn` itself fails, its error
  // goes to `on_error` when given (handled there), else it becomes the
  // device's sticky error.
  absl::Status HostCallback(
      std::function<absl::Status()> fn,
      std::function<void(absl::Status)> on_error = nullptr);

  // Records everything enqueued on (and waited for by) the stream so far.
  absl::Status RecordEvent(Event* event);
  // Later work on this stream waits for the event's work / for everything
  // currently enqueued on `other` (and what `other` waits for).
  absl::Status WaitForEvent(Event* event);
  absl::Status WaitForStream(Stream* other);

  // Block until everything enqueued on the stream is done. Returns the
  // device's sticky error, if any (see the file comment).
  absl::Status Synchronize();

  // Testing only: the next committed command buffer is treated as failed
  // with `error`. Its work runs but its timeline signal does not; its
  // completion handler records the error and force-signals it, as for a
  // buffer the GPU aborted.
  void FailNextCommandBufferForTesting(absl::Status error);
  // Testing only: the last committed buffer's timeline value, and the
  // timeline's signaled value now.
  std::pair<uint64_t, uint64_t> FenceForTesting();
  // Commit the device's open command buffer without waiting.
  absl::Status Flush();

  // Command buffer batching. Each command buffer costs a fixed amount of
  // CPU and GPU-scheduler time (hundreds of microseconds on an M3), so
  // workloads made of many tiny kernels (scans, while loops) need many
  // dispatches per buffer. Policy: commit when the GPU has nothing of ours
  // left to run and at least kEarlyCommitOps ops are encoded (keeps the GPU
  // fed with low latency), or when kMaxOpsPerCommandBuffer ops,
  // kMaxThreadsPerCommandBuffer thread-equivalents or kMaxFlopsPerCommandBuffer
  // GEMM flops are encoded (the latter two keep each buffer far from the GPU
  // watchdog).
  static constexpr int kMaxOpsPerCommandBuffer = 1024;
  // Waiting for the GPU to run out limits early commits to about one per
  // GPU round trip (~150 us), so loops of tiny kernels make few buffers
  // (a 2000-step scan: ~8 per call).
  static constexpr int kEarlyCommitOps = 16;
  // Copies and uniform fills up to this size run as compute kernels.
  static constexpr uint64_t kComputeCopyMaxBytes = 16ull << 20;
  // Also commit once this many threads have been dispatched into one command
  // buffer (roughly tens of milliseconds of GPU work), so a batch of heavy
  // kernels cannot approach the GPU watchdog timeout, especially under
  // contention from other processes.
  static constexpr uint64_t kMaxThreadsPerCommandBuffer = 1ull << 27;
  // GEMMs launch few threads for their work, so they are charged their
  // flop-equivalents (Launch/EncodeExternal `flops`) against this budget
  // instead. 2e11 is <= 0.1 s of GEMM on the M3 this was measured on
  // (2.0-3.1 TFLOPS over f32 MPS and f16/bf16 steel shapes, 64 flops per
  // byte for bandwidth-bound ones; see blas::GemmWork). That leaves a >= 5x
  // margin for the slowest supported GPU (a base M1, 2-3x slower when
  // throttled) and for shapes steel runs less efficiently, well under the
  // watchdog. An op that would take the buffer over it starts a new buffer;
  // a single larger GEMM gets a buffer of its own.
  static constexpr uint64_t kMaxFlopsPerCommandBuffer = 200'000'000'000ull;

 private:
  friend class Device;
  Stream(Device* device, MTL::SharedEvent* host_fence);

  absl::Status LaunchWithArgumentBuffer(const Kernel& kernel,
                                        Dim3 threadgroups, Dim3 threads,
                                        absl::Span<const KernelArg> args,
                                        uint32_t threadgroup_memory_bytes,
                                        uint64_t flops);
  // Before encoding a GPU op: waits on the host for the host tasks the
  // stream waits for (not holding the queue lock), then clears its waits.
  // Caller holds mu_.
  absl::Status BeginGpuOpLocked();
  // After encoding an op into the open buffer (caller holds mu_ and the
  // queue lock): the stream's position is that buffer; then the batching
  // policy.
  absl::Status FinishGpuOpLocked(uint64_t work, uint64_t flops = 0);
  // Everything enqueued on or waited for by the stream. Caller holds mu_.
  WorkPoint CurrentLocked() const;
  // True when CurrentLocked() is done (host transfers then run inline).
  // Caller holds mu_.
  bool IdleLocked();
  // FailedPreconditionError when called from a host task (or its on_error)
  // of this stream; checked before taking mu_.
  absl::Status RefuseOwnHostTask() const;

  Device* device_;
  MTL::SharedEvent* host_fence_;  // owned by the device
  std::mutex mu_;
  uint64_t gpu_pos_ = 0;    // timeline value of the stream's last GPU op
  WorkPoint waits_;         // what the next op must wait for
  uint64_t host_value_ = 0;  // last host task value issued

  // Host work ordered on the stream: each task waits for `after`, runs,
  // then signals `value` on the host fence.
  struct HostTask {
    WorkPoint after;
    std::function<absl::Status()> fn;
    std::function<void(absl::Status)> on_error;
    uint64_t value;
    uint64_t ticket = 0;  // Device::BeginWork
  };
  void WorkerLoop();
  std::thread worker_;
  std::mutex work_mu_;
  std::condition_variable work_cv_;
  std::deque<HostTask> work_;
  bool stop_ = false;
};

// Device::memory_stats() of the live device `ordinal` (false: none). For
// PJRT_Device_MemoryStats (pjrt/metal_pjrt_api.cc).
bool MemoryStatsOf(int ordinal, Device::MemoryStats* out);

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_METAL_RUNTIME_H_
