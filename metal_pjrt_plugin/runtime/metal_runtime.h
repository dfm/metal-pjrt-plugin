// Thin C++ runtime over Metal (via metal-cpp), independent of XLA (it depends
// only on absl) so it can be unit-tested standalone. The StreamExecutor adapter translates this API into
// XLA's abstractions.
//
// Model:
//   Device   one MTL::Device + allocator bookkeeping + library/pipeline caches
//   Stream   one MTL::CommandQueue; work is encoded into an open command buffer
//            that is committed at sync points or when an op budget is reached
//   Event    a value on an MTL::SharedEvent (timeline semaphore)
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
#ifndef METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_
#define METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_

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
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

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
  bool unified_memory = true;
  uint64_t max_buffer_length = 0;          // largest single MTL::Buffer
  uint64_t recommended_working_set = 0;    // bytes
  uint32_t max_threads_per_threadgroup = 1024;
  uint32_t threadgroup_memory_length = 32768;
  uint32_t simd_width = 32;
  bool supports_metal4 = false;            // Apple9 family (M3+)
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
// with more buffer arguments than Metal's argument table holds. Must match
// codegen::kArgumentBufferMarker (codegen/msl_kernel.h).
inline constexpr char kArgumentBufferMarker[] = "// xla_metal_argbuffer";

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

class Kernel {
 public:
  Kernel(MTL::ComputePipelineState* pso,
         std::shared_ptr<const KernelIdentity> identity)
      : pso_(pso), identity_(std::move(identity)) {}
  ~Kernel();
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  MTL::ComputePipelineState* pso() const { return pso_; }
  const std::string& name() const { return identity_->name; }
  const std::string& key() const { return identity_->key; }
  const std::shared_ptr<const KernelIdentity>& identity() const {
    return identity_;
  }
  // See kArgumentBufferMarker. Launch then packs all (buffer) arguments into
  // one setBytes payload of GPU addresses and marks the buffers resident.
  bool uses_argument_buffer() const { return uses_argument_buffer_; }
  void set_uses_argument_buffer(bool v) { uses_argument_buffer_ = v; }
  // The source's declared [[max_total_threads_per_threadgroup]] (0: none).
  // Release Metal does not check dispatches against it (a larger threadgroup
  // is undefined behaviour on the GPU), so max_total_threads_per_threadgroup
  // includes it and launches above it are refused before encoding.
  void set_declared_max_threads(uint32_t n) { declared_max_threads_ = n; }
  uint32_t max_total_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;
  uint32_t static_threadgroup_memory_length() const;

 private:
  MTL::ComputePipelineState* pso_;
  std::shared_ptr<const KernelIdentity> identity_;
  bool uses_argument_buffer_ = false;
  uint32_t declared_max_threads_ = 0;
};

class Event;
class Stream;

class Device {
 public:
  static absl::StatusOr<std::unique_ptr<Device>> Create(int ordinal);
  static int VisibleDeviceCount();
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceInfo& info() const { return info_; }

  // Memory this process may hold (live + cached): half of physical RAM,
  // capped by the GPU's recommended working set, times
  // JAX_OPENMETAL_MEMORY_FRACTION (default 1).
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
  // the system memory guard passes (dropping the whole cache and retrying
  // once); otherwise RESOURCE_EXHAUSTED. Cached buffers are released when
  // unused for kCacheIdleRelease, and all at once on a system memory-pressure
  // warning (after which frees release until the level is normal again), so
  // an idle process gives its memory back. Releasing a buffer that in-flight
  // work still uses is safe: command buffers retain what they bind.
  absl::StatusOr<Allocation> Allocate(uint64_t size);
  absl::Status Deallocate(void* ptr);
  // Resolve a raw pointer (possibly interior) to its buffer and offset.
  absl::StatusOr<BufferRef> Resolve(const void* ptr) const;
  // Live bytes (allocated, not freed; buffer lengths).
  uint64_t allocated_bytes() const;
  // Bumped whenever a buffer is really released (not when one is recycled
  // through the cache); lets streams cache Resolve results.
  uint64_t allocation_generation() const {
    return allocation_generation_.load(std::memory_order_acquire);
  }
  static constexpr std::chrono::seconds kCacheIdleRelease{2};
  struct MemoryStats {
    uint64_t live_bytes = 0;
    uint64_t cached_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    int pressure = 0;  // 0 normal, 1 warning, 2 critical
  };
  MemoryStats memory_stats() const;
  // What the DISPATCH_SOURCE_TYPE_MEMORYPRESSURE handler calls (level as
  // above); tests call it through metal_pjrt_memory_pressure().
  void OnMemoryPressure(int level);
  // Releases cached buffers freed at least `min_idle` ago (all with 0).
  void TrimCache(std::chrono::steady_clock::duration min_idle);

  // Compile Metal Shading Language source into a library, cached by content.
  // The library is owned by the device's cache.
  absl::StatusOr<MTL::Library*> CompileLibrary(const std::string& msl_source);
  // Create (or fetch cached) pipeline state for `function` in `library`.
  absl::StatusOr<std::unique_ptr<Kernel>> CreateKernel(
      MTL::Library* library, const std::string& function,
      bool builtin = false);

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
  // timed out (none when that buffer only waited on another stream; built-in
  // kernels are listed separately and never blamed). Kernels seen in
  // `quarantine_strikes()` or more resets since boot are refused by
  // CreateKernel until a reboot or until the log is cleared
  // (scripts/gpu_health.py --clear). A kernel whose source changes gets a
  // new key; a fix elsewhere (runtime, launch dimensions) needs --clear.
  // The state directory is METAL_PJRT_STATE_DIR or ~/.cache/openmetal (whose
  // reset log starts as a copy of ~/.cache/jax_metal's, if any);
  // METAL_PJRT_QUARANTINE_STRIKES sets the threshold (0 disables).
  void RecordReset(absl::string_view cause,
                   absl::Span<const std::shared_ptr<const KernelIdentity>>
                       kernels);
  const std::string& state_dir() const { return state_dir_; }
  int quarantine_strikes() const { return quarantine_strikes_; }
  // Resets recorded since the machine booted (by any process).
  int resets_since_boot() const { return resets_since_boot_; }

  // Hold-rule counters (see Stream::Commit): command buffers whose commit
  // waited on the host for a host task, and command buffers committed while
  // still waiting on an unfinished host task (must stay 0).
  uint64_t host_task_holds() const { return host_task_holds_.load(); }
  uint64_t unsignaled_host_task_waits_committed() const {
    return unsignaled_host_task_waits_committed_.load();
  }

 private:
  Device() = default;
  int ordinal_ = 0;
  MTL::Device* device_ = nullptr;
  DeviceInfo info_;

  mutable std::mutex mu_;
  // Keyed by start address; value is the buffer and its length.
  std::map<uintptr_t, std::pair<MTL::Buffer*, uint64_t>> allocations_;
  uint64_t allocated_bytes_ = 0;
  std::atomic<uint64_t> allocation_generation_{0};
  uint64_t memory_budget_ = 0;
  // The free-buffer cache (see Allocate), guarded by mu_: least recently
  // freed first, and indexed by length.
  struct CachedBuffer {
    MTL::Buffer* buffer;
    uint64_t size;
    std::chrono::steady_clock::time_point freed;
  };
  std::list<CachedBuffer> cache_;
  std::multimap<uint64_t, std::list<CachedBuffer>::iterator> cache_by_size_;
  uint64_t cached_bytes_ = 0;
  uint64_t cache_hits_ = 0;
  uint64_t cache_misses_ = 0;
  int pressure_ = 0;
  // Removes `it` from the cache and appends its buffer to `out`.
  void EvictLocked(std::list<CachedBuffer>::iterator it,
                   std::vector<MTL::Buffer*>* out);
  // Releases buffers taken out of the cache (outside mu_).
  void ReleaseBuffers(const std::vector<MTL::Buffer*>& buffers);
  // libdispatch sources on memory_queue_: the memory-pressure source and a
  // timer running TrimCache(kCacheIdleRelease), resumed only while the cache
  // is not empty (trim_armed_, guarded by mu_).
  void* memory_queue_ = nullptr;     // dispatch_queue_t
  void* pressure_source_ = nullptr;  // dispatch_source_t
  void* trim_timer_ = nullptr;       // dispatch_source_t
  bool trim_armed_ = false;
  void StartMemorySources();
  absl::Status error_ = absl::OkStatus();
  std::unordered_map<std::string, MTL::Library*> library_cache_;  // key: source hash
  std::unordered_map<std::string, MTL::ComputePipelineState*> pso_cache_;
  std::unordered_map<MTL::Library*, std::string> library_keys_;  // hash
  std::mutex builtin_mu_;
  std::unique_ptr<Kernel> builtin_kernels_[4];
  // Reset log state (see RecordReset); strikes_ is guarded by mu_.
  void LoadResetLog();
  std::string state_dir_;
  int quarantine_strikes_ = 2;
  int resets_since_boot_ = 0;
  std::unordered_map<std::string, int> strikes_;  // kernel key -> resets
  std::string build_;  // hex ImageUuid, recorded with each reset
  friend class Stream;
  friend class Event;
  // Committed command buffers whose completion handler has not run yet
  // (retained, with their stream fence, retained, and the fence value they
  // end by signaling; `injected` is FailNextCommandBufferForTesting's error,
  // which counts as the buffer's status). Completion handlers remove
  // theirs; they never block.
  struct InFlight {
    MTL::CommandBuffer* cb;
    MTL::SharedEvent* fence;
    uint64_t value;
    absl::Status injected;
  };
  std::mutex in_flight_mu_;
  std::vector<InFlight> in_flight_;
  void AddInFlight(MTL::CommandBuffer* cb, MTL::SharedEvent* fence,
                   uint64_t value, absl::Status injected);
  void RemoveInFlight(MTL::CommandBuffer* cb);
  // Called by host waiters after they saw a signal: sets error() if any
  // in-flight command buffer has failed. A buffer whose fence value is
  // signaled has run to its end, so with `wait` its final status is waited
  // for (Metal sets it before running handlers); without, returns false if
  // some such status is not final yet. This covers every buffer the waiter
  // depends on, on any queue, without waiting for handlers (which Metal
  // does not order across queues): a buffer that ran its signals signaled
  // its fence first (see Stream::Commit), and one that did not is
  // force-signaled by its handler after the error is recorded.
  bool CheckInFlight(bool wait);
  std::atomic<uint64_t> host_task_holds_{0};
  std::atomic<uint64_t> unsignaled_host_task_waits_committed_{0};
};

// Timeline-semaphore style event: `Record` on a stream bumps and signals a
// value; waiting (on host or another stream) targets that value.
class Event {
 public:
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  // True once the last recorded value has been signaled (also on failure).
  bool IsComplete() const;
  // Non-blocking: false while pending, true once complete, or the device's
  // sticky error.
  absl::StatusOr<bool> Poll();
  // Block the host until complete. Safe from host tasks: a value is
  // published only once its signaling buffer is committed, and committed
  // buffers never wait for unfinished host tasks.
  absl::Status WaitOnHost();

 private:
  friend class Device;
  friend class Stream;
  Event(Device* device, MTL::SharedEvent* ev) : device_(device), event_(ev) {}
  Device* device_;
  MTL::SharedEvent* event_;
  // Last recorded value whose signaling buffer is committed; 0 means never
  // recorded. next_value_ is the last value handed out by RecordEvent.
  uint64_t value_ = 0;
  uint64_t next_value_ = 0;
  std::mutex mu_;
};

class Stream {
 public:
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  // Kernel launch. threadgroups = grid in threadgroup units, threads = per
  // threadgroup. Buffer args are resolved to (buffer, offset) here.
  absl::Status Launch(const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
                      absl::Span<const KernelArg> args,
                      uint32_t threadgroup_memory_bytes = 0);

  // Metal's per-stage buffer argument table has 31 slots. Kernels using an
  // argument buffer (Kernel::uses_argument_buffer) take up to
  // kMaxArgumentBufferArgs buffer arguments (4 KB of setBytes payload).
  static constexpr size_t kMaxBufferArgs = 31;
  static constexpr size_t kMaxArgumentBufferArgs = 4096 / sizeof(uint64_t);

  // Encode work produced outside this runtime (e.g. Metal Performance
  // Shaders) into the stream's open command buffer, in order with all other
  // stream work. Ends the current compute encoder, then calls `encode` with the
  // raw MTL::CommandBuffer* (as void*, so Objective-C++ callers can bridge it
  // to id<MTLCommandBuffer>). `encode` must create, use and end its own
  // encoders and must not commit the buffer or call back into this stream (the
  // stream lock is held). Counts as one op toward kMaxOpsPerCommandBuffer.
  absl::Status EncodeExternal(
      std::function<absl::Status(void* mtl_command_buffer)> encode);

  // Device-to-device copy and fill. Copies and single-byte fills use a blit
  // encoder; other fills use the device's built-in fill kernel.
  absl::Status MemcpyDeviceToDevice(void* dst, const void* src, uint64_t size);
  absl::Status Memset8(void* dst, uint8_t value, uint64_t size);
  absl::Status Memset32(void* dst, uint32_t value, uint64_t size);

  // Host transfers are ordered on the stream: they run in a host callback once
  // prior work completes, and later stream work waits for them. With unified
  // memory these are plain memcpys.
  absl::Status MemcpyHostToDevice(void* dst, const void* src, uint64_t size);
  absl::Status MemcpyDeviceToHost(void* dst, const void* src, uint64_t size);

  // Run `fn` on the host once all previously enqueued work has completed;
  // subsequent stream work waits for it to finish. GPU work that waits for a
  // host task is committed only after the task has run (see Commit), so a
  // slow task never counts against the GPU watchdog. A task must not wait
  // for its own stream: Synchronize, Event::WaitOnHost and commits that would
  // wait for the running task (or later work on its stream) return
  // FailedPreconditionError instead of deadlocking.
  //
  // After a failure (the device's sticky error), a task with `on_error` does
  // not run and `on_error` gets the error; one without it still runs (XLA's
  // plain callbacks free memory or complete transfers). If `fn` itself
  // fails, its error goes to `on_error` when given (handled there), else it
  // becomes the device's sticky error.
  absl::Status HostCallback(
      std::function<absl::Status()> fn,
      std::function<void(absl::Status)> on_error = nullptr);

  absl::Status RecordEvent(Event* event);
  absl::Status WaitForEvent(Event* event);
  // Make this stream wait for everything currently enqueued on `other`,
  // including its host tasks and the waits it has not encoded yet.
  absl::Status WaitForStream(Stream* other);

  // Commit any open work and block until the stream is idle. Returns the
  // device's sticky error, if any (see the file comment).
  absl::Status Synchronize();

  // Testing only: the next committed command buffer is treated as failed
  // with `error`. Its work runs but its signals do not; its completion
  // handler records the error and force-signals them, as for a buffer the
  // GPU aborted.
  void FailNextCommandBufferForTesting(absl::Status error);
  // Testing only: the fence value the last committed command buffer
  // signals, and the fence's signaled value now.
  std::pair<uint64_t, uint64_t> FenceForTesting();
  // Commit any open work without waiting.
  absl::Status Flush();

  // Command buffer batching. Each command buffer costs a fixed amount of
  // CPU and GPU-scheduler time (hundreds of microseconds on an M3), so
  // workloads made of many tiny kernels (scans, while loops) need many
  // dispatches per buffer. Policy: commit when the GPU has nothing of ours
  // left to run and at least kEarlyCommitOps ops are encoded (keeps the GPU
  // fed with low latency), or when kMaxOpsPerCommandBuffer ops or
  // kMaxThreadsPerCommandBuffer thread-equivalents of work are encoded (the
  // latter keeps each buffer far from the GPU watchdog). METAL_PJRT_MAX_OPS
  // overrides the op cap.
  static constexpr int kMaxOpsPerCommandBuffer = 1024;
  static constexpr int kEarlyCommitOps = 16;
  // Early commits are also paced in time: submitting a command buffer costs
  // the driver ~150 us of a dedicated thread, which was the bottleneck for
  // loops of tiny kernels (700 buffers per call). At most one early commit
  // per kEarlyCommitIntervalUs; explicit syncs and the caps above still
  // commit immediately. METAL_PJRT_EARLY_COMMIT_US overrides.
  static constexpr int kEarlyCommitIntervalUs = 500;
  // Copies and uniform fills up to this size run as compute kernels.
  static constexpr uint64_t kComputeCopyMaxBytes = 16ull << 20;
  // Also commit once this many threads have been dispatched into one command
  // buffer (roughly tens of milliseconds of GPU work), so a batch of heavy
  // kernels cannot approach the GPU watchdog timeout, especially under
  // contention from other processes.
  static constexpr uint64_t kMaxThreadsPerCommandBuffer = 1ull << 27;
  // Work charged for an op whose cost we cannot see (EncodeExternal, e.g. an
  // MPS GEMM): at most 32 such ops per command buffer, as before batching.
  static constexpr uint64_t kExternalOpWork = kMaxThreadsPerCommandBuffer / 32;

 private:
  friend class Device;
  Stream(Device* device, MTL::CommandQueue* queue, MTL::SharedEvent* fence);

  absl::Status LaunchWithArgumentBuffer(const Kernel& kernel,
                                        Dim3 threadgroups, Dim3 threads,
                                        absl::Span<const KernelArg> args,
                                        uint32_t threadgroup_memory_bytes);
  // Encoders for already-resolved work. Caller holds mu_.
  absl::Status EncodeLaunch(const Kernel& kernel, Dim3 threadgroups,
                            Dim3 threads, absl::Span<const KernelArg> args,
                            absl::Span<const BufferRef> refs,
                            uint32_t threadgroup_memory_bytes);
  absl::Status EncodeLaunchWithArgumentBuffer(
      const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
      absl::Span<const uint64_t> addrs,
      absl::Span<MTL::Buffer* const> resident,
      uint32_t threadgroup_memory_bytes);
  absl::Status EncodeBuiltin(Device::Builtin kind,
                             absl::Span<const BufferRef> buffers,
                             const void* bytes, size_t bytes_len, uint64_t n);
  absl::Status EncodeCopy(BufferRef dst, BufferRef src, uint64_t size);
  absl::Status EncodeFill(BufferRef dst, uint32_t pattern, int pattern_bytes,
                          uint64_t size);
  // Device::Resolve with a small per-stream cache of recently used
  // allocations (invalidated whenever anything is deallocated). Caller holds
  // mu_.
  absl::StatusOr<BufferRef> ResolveCached(const void* ptr);
  // Account one encoded op of `work` thread-equivalents and commit the
  // command buffer if the batching policy says so. Caller holds mu_.
  absl::Status FinishOp(uint64_t work);

  // Ensure an open command buffer/encoder exist.
  absl::Status EnsureCommandBuffer();
  absl::Status EnsureComputeEncoder();
  void EndEncoder();
  // Commit the current command buffer (if any). Hold rule: if the buffer
  // waits on a host-task value that is not signaled yet, first wait for it
  // on the host (bounded, like every host wait), so no committed buffer ever
  // sits on the GPU waiting for host work. Host-task workers never take mu_,
  // so waiting here while holding it cannot deadlock with them.
  absl::Status Commit();
  Device* device_;
  MTL::CommandQueue* queue_;
  MTL::SharedEvent* fence_;      // private timeline for this stream
  uint64_t fence_value_ = 0;     // last value signaled on fence_
  uint64_t last_committed_fence_value_ = 0;
  MTL::CommandBuffer* cmd_ = nullptr;
  MTL::ComputeCommandEncoder* enc_ = nullptr;
  int ops_in_cmd_ = 0;
  uint64_t threads_in_cmd_ = 0;
  std::chrono::steady_clock::time_point last_commit_time_{};
  // State shared with completion handlers, which may run after the stream is
  // gone (~Stream abandons in-flight buffers after a failure).
  struct CompletionState {
    // Committed command buffers whose completion handler has not run yet.
    std::atomic<int> pending{0};
  };
  // FailNextCommandBufferForTesting.
  absl::Status inject_error_;
  const std::shared_ptr<CompletionState> completion_ =
      std::make_shared<CompletionState>();
  // ResolveCached state (guarded by mu_).
  struct CachedRange {
    uintptr_t base = 0;
    uint64_t size = 0;
    MTL::Buffer* buffer = nullptr;
  };
  static constexpr int kResolveCacheSize = 8;
  CachedRange resolve_cache_[kResolveCacheSize];
  int resolve_cache_next_ = 0;
  uint64_t resolve_cache_generation_ = ~0ull;
  // Committed but possibly still executing command buffers (retained).
  // Synchronize waits for their completion, not just the fence signal, so
  // callers may free resources immediately afterwards.
  std::vector<MTL::CommandBuffer*> in_flight_;
  std::mutex mu_;
  // Events the open command buffer signals, encoded by Commit after the
  // fence signal (so a signaled event means the buffer has run to its end;
  // see Device::CheckInFlight). On failure they are force-signaled so
  // waiters wake up instead of hanging.
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> pending_signals_;
  // Waits encoded into the open command buffer (hold rule, diagnostics).
  struct PendingWait {
    MTL::SharedEvent* event;
    uint64_t value;
    const char* kind;
    // The value is signaled by a host task (of any stream), not by GPU work.
    bool host_task = false;
  };
  std::vector<PendingWait> pending_waits_;
  // Waits requested (WaitForEvent/WaitForStream/HostCallback) but not yet
  // encoded. A command buffer that only waits still counts its waiting time
  // against the GPU watchdog, so waits are encoded lazily, right before the
  // next GPU work on this stream; a host-side Synchronize or host task waits
  // for them on the host instead. Events are not retained: their owners
  // (Event objects, other streams) outlive the stream, and stream fences are
  // released only in ~Stream after work has drained.
  std::vector<PendingWait> deferred_waits_;
  // Encode deferred waits into the open command buffer. Caller holds mu_ and
  // has an open command buffer; any open encoder is ended first.
  void FlushDeferredWaits();
  // Kernels encoded into the open command buffer (for the failure
  // diagnostics and the reset log).
  std::vector<std::shared_ptr<const KernelIdentity>> pending_kernels_;
  // Host work ordered on the stream: each task waits for the fence to reach
  // wait_value, runs, then signals signal_value so later GPU work proceeds.
  struct HostTask {
    uint64_t wait_value;
    std::function<absl::Status()> fn;
    std::function<void(absl::Status)> on_error;
    uint64_t signal_value;
    // Cross-stream waits that were pending when the task was enqueued: the
    // task also waits for these (retained events).
    std::vector<std::pair<MTL::SharedEvent*, uint64_t>> extra_waits;
  };
  void WorkerLoop();
  std::thread worker_;
  std::mutex work_mu_;
  std::condition_variable work_cv_;
  std::deque<HostTask> work_;
  bool stop_ = false;
};

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_
