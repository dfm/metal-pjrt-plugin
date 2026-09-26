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
//   FailedPreconditionError the stream is in the error state after an earlier
//                           GPU failure
#ifndef METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_
#define METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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

// What identifies a kernel across processes: its function name and a key
// made of the hash of its MSL source plus the function name. Shared by the
// Kernel and by the streams' per-command-buffer records (so recording a
// launch costs a reference count, not a string copy).
struct KernelIdentity {
  std::string name;
  std::string key;
};

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
  uint32_t max_total_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;
  uint32_t static_threadgroup_memory_length() const;

 private:
  MTL::ComputePipelineState* pso_;
  std::shared_ptr<const KernelIdentity> identity_;
  bool uses_argument_buffer_ = false;
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

  // Memory the allocator pool may grow to in this process. Computed at
  // creation from what the system actually had free (minus a reserve for the
  // OS and other processes), capped by the GPU's recommended working set.
  uint64_t memory_budget() const { return memory_budget_; }

  // Set when the GPU had to be reset (watchdog timeout, access revoked,
  // device removed). Sticky for the process: continuing to submit work to a
  // GPU that is being reset makes recovery less likely, and CUDA treats the
  // equivalent as a fatal context error too.
  absl::Status lost_status() const;
  void MarkLost(const absl::Status& why);
  int ordinal() const { return ordinal_; }
  MTL::Device* mtl() const { return device_; }

  // Memory. Allocations are shared-storage MTL::Buffers; freeing an address
  // that was not returned by Allocate is an error.
  absl::StatusOr<Allocation> Allocate(uint64_t size);
  absl::Status Deallocate(void* ptr);
  // Resolve a raw pointer (possibly interior) to its buffer and offset.
  absl::StatusOr<BufferRef> Resolve(const void* ptr) const;
  uint64_t allocated_bytes() const;
  // Bumped by every Deallocate; lets streams cache Resolve results.
  uint64_t allocation_generation() const {
    return allocation_generation_.load(std::memory_order_acquire);
  }

  // Compile Metal Shading Language source into a library, cached by content.
  // The library is owned by the device's cache.
  absl::StatusOr<MTL::Library*> CompileLibrary(const std::string& msl_source);
  // Create (or fetch cached) pipeline state for `function` in `library`.
  absl::StatusOr<std::unique_ptr<Kernel>> CreateKernel(
      MTL::Library* library, const std::string& function);

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream();
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent();

  // Built-in fill kernels (32-bit and byte granularity), compiled on first
  // use. Used for memsets whose pattern is not a single repeated byte.
  absl::StatusOr<const Kernel*> FillKernel(bool word_granular);

  // GPU reset log and kernel quarantine. A watchdog timeout resets the GPU
  // for every process, and after a few resets the driver leaves the GPU slow
  // until a reboot, so the same bug must not be allowed to reset it again
  // and again. Every reset observed by this process is appended to
  // <state dir>/gpu_resets.jsonl with the kernels that were in the command
  // buffer that timed out (none when that buffer only waited on another
  // stream). Kernels seen in `quarantine_strikes()` or more resets are refused
  // by CreateKernel until the log is cleared (scripts/gpu_health.py --clear).
  // The state directory is METAL_PJRT_STATE_DIR or ~/.cache/jax_metal;
  // METAL_PJRT_QUARANTINE_STRIKES sets the threshold (0 disables).
  void RecordReset(absl::string_view cause,
                   absl::Span<const std::shared_ptr<const KernelIdentity>>
                       kernels);
  const std::string& state_dir() const { return state_dir_; }
  int quarantine_strikes() const { return quarantine_strikes_; }
  // Resets recorded since the machine booted (by any process).
  int resets_since_boot() const { return resets_since_boot_; }

 private:
  Device() = default;
  int ordinal_ = 0;
  MTL::Device* device_ = nullptr;
  DeviceInfo info_;

  mutable std::mutex mu_;
  // Keyed by start address; value is the buffer and its size.
  std::map<uintptr_t, std::pair<MTL::Buffer*, uint64_t>> allocations_;
  uint64_t allocated_bytes_ = 0;
  std::atomic<uint64_t> allocation_generation_{0};
  uint64_t memory_budget_ = 0;
  absl::Status lost_ = absl::OkStatus();
  std::vector<Stream*> live_streams_;  // guarded by mu_; for diagnostics
 public:
  void RegisterStream(Stream* s);
  void UnregisterStream(Stream* s);
  // Log every live stream's DebugState (used when a command buffer times out).
  void DumpStreams();
 private:
  std::unordered_map<std::string, MTL::Library*> library_cache_;  // key: source hash
  std::unordered_map<std::string, MTL::ComputePipelineState*> pso_cache_;
  std::unordered_map<MTL::Library*, std::string> library_keys_;  // hash
  std::mutex fill_mu_;
  std::unique_ptr<Kernel> fill32_kernel_;
  std::unique_ptr<Kernel> fill8_kernel_;
  // Reset log state (see RecordReset); strikes_ is guarded by mu_.
  void LoadResetLog();
  std::string state_dir_;
  int quarantine_strikes_ = 2;
  int resets_since_boot_ = 0;
  std::unordered_map<std::string, int> strikes_;  // kernel key -> resets
};

// Timeline-semaphore style event: `Record` on a stream bumps and signals a
// value; waiting (on host or another stream) targets that value.
class Event {
 public:
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  // True once the last recorded value has been signaled.
  bool IsComplete() const;
  // Block the host until complete.
  absl::Status WaitOnHost();

 private:
  friend class Device;
  friend class Stream;
  Event(Device* device, MTL::SharedEvent* ev) : device_(device), event_(ev) {}
  Device* device_;
  MTL::SharedEvent* event_;
  uint64_t value_ = 0;  // last recorded value; 0 means never recorded
  std::mutex mu_;
};

// A recorded, replayable sequence of stream work: the software command
// buffer behind StreamExecutor's CommandBuffer on Metal. Commands keep their
// buffer arguments resolved to (MTL::Buffer, offset) so Stream::Replay only
// encodes; resolution is redone lazily when the device's allocation
// generation changes (something was freed since) or when a command is
// updated with new arguments. Commands replay in recording order, with the
// stream's usual serial ordering between them.
//
// Measured on an M3 (runtime/icb_spike.cc): direct encoding costs about
// 0.1-0.2 us per dispatch, so replaying this way removes the 30-40 us of
// per-kernel host work XLA's thunks and our argument packing cost, without the
// 1-4 us of extra GPU time per dispatch a Metal indirect command buffer with
// per-command barriers was measured to add.
class CommandList {
 public:
  explicit CommandList(Device* device) : device_(device) {}
  CommandList(const CommandList&) = delete;
  CommandList& operator=(const CommandList&) = delete;

  // Each Add* returns the new command's index; Update* replaces the
  // parameters of an existing command of the same kind. Arguments are
  // validated and resolved immediately.
  absl::StatusOr<size_t> AddLaunch(const Kernel* kernel, Dim3 threadgroups,
                                   Dim3 threads,
                                   absl::Span<const KernelArg> args,
                                   uint32_t threadgroup_memory_bytes = 0);
  absl::Status UpdateLaunch(size_t index, const Kernel* kernel,
                            Dim3 threadgroups, Dim3 threads,
                            absl::Span<const KernelArg> args,
                            uint32_t threadgroup_memory_bytes = 0);
  absl::StatusOr<size_t> AddCopy(void* dst, const void* src, uint64_t size);
  absl::Status UpdateCopy(size_t index, void* dst, const void* src,
                          uint64_t size);
  // `pattern` is the fill value broadcast to 32 bits; `pattern_bytes` (1, 2
  // or 4) is the width of the original pattern.
  absl::StatusOr<size_t> AddFill(void* dst, uint32_t pattern,
                                 int pattern_bytes, uint64_t size);
  absl::Status UpdateFill(size_t index, void* dst, uint32_t pattern,
                          int pattern_bytes, uint64_t size);
  size_t AddEmpty();

  size_t size() const { return commands_.size(); }
  Device* device() const { return device_; }
  std::string ToString() const;

 private:
  friend class Stream;
  struct Command {
    enum class Kind { kLaunch, kCopy, kFill, kEmpty };
    Kind kind = Kind::kEmpty;
    // Launch.
    const Kernel* kernel = nullptr;
    Dim3 threadgroups{1, 1, 1};
    Dim3 threads{1, 1, 1};
    uint32_t threadgroup_memory_bytes = 0;
    std::vector<KernelArg> args;
    std::vector<BufferRef> refs;           // per argument (buffers only)
    std::vector<uint64_t> addrs;           // argument-buffer kernels
    std::vector<MTL::Buffer*> resident;    // argument-buffer kernels
    // Copy / fill.
    void* dst = nullptr;
    const void* src = nullptr;
    uint64_t size = 0;
    uint32_t pattern = 0;
    int pattern_bytes = 0;
    BufferRef dst_ref;
    BufferRef src_ref;
  };
  absl::Status Resolve(Command& c, size_t index);
  // Re-resolve every command if anything was freed since the last time.
  absl::Status ResolveIfStale();
  absl::Status CheckIndex(size_t index, Command::Kind kind) const;

  Device* device_;
  std::vector<Command> commands_;
  uint64_t resolved_generation_ = ~0ull;
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

  // Encode every command of `list` into this stream, in order. The list must
  // belong to this stream's device.
  absl::Status Replay(CommandList& list);

  // Host transfers are ordered on the stream: they run in a host callback once
  // prior work completes, and later stream work waits for them. With unified
  // memory these are plain memcpys.
  absl::Status MemcpyHostToDevice(void* dst, const void* src, uint64_t size);
  absl::Status MemcpyDeviceToHost(void* dst, const void* src, uint64_t size);

  // Run `fn` on the host once all previously enqueued work has completed;
  // subsequent stream work waits for it to finish.
  absl::Status HostCallback(std::function<void()> fn);

  absl::Status RecordEvent(Event* event);
  absl::Status WaitForEvent(Event* event);
  // Make this stream wait for everything currently enqueued on `other`.
  absl::Status WaitForStream(Stream* other);

  // Commit any open work and block until the stream is idle. Returns the GPU
  // error (InternalError) the first time a failure is observed and
  // FailedPreconditionError on every later call: the stream stays in the error
  // state and also refuses new work.
  absl::Status Synchronize();
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
  // Encoders for already-resolved work (shared by the direct API and Replay).
  // Caller holds mu_.
  absl::Status EncodeLaunch(const Kernel& kernel, Dim3 threadgroups,
                            Dim3 threads, absl::Span<const KernelArg> args,
                            absl::Span<const BufferRef> refs,
                            uint32_t threadgroup_memory_bytes);
  absl::Status EncodeLaunchWithArgumentBuffer(
      const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
      absl::Span<const uint64_t> addrs,
      absl::Span<MTL::Buffer* const> resident,
      uint32_t threadgroup_memory_bytes);
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
  // Commit the current command buffer (if any).
  absl::Status Commit();
  // Signal/wait on the stream's private timeline (used for host callbacks and
  // cross-stream ordering).
  uint64_t SignalFence();

  Device* device_;
  MTL::CommandQueue* queue_;
  MTL::SharedEvent* fence_;      // private timeline for this stream
  uint64_t fence_value_ = 0;     // last value signaled on fence_
  uint64_t last_committed_fence_value_ = 0;
  MTL::CommandBuffer* cmd_ = nullptr;
  MTL::ComputeCommandEncoder* enc_ = nullptr;
  int ops_in_cmd_ = 0;
  uint64_t threads_in_cmd_ = 0;
  // Committed command buffers whose completion handler has not run yet.
  std::atomic<int> gpu_pending_{0};
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
  // Events signaled by the open command buffer; on GPU error they are
  // force-signaled so host waiters wake up instead of hanging.
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> pending_signals_;
  // Waits encoded into the open command buffer, kept for diagnostics.
  struct PendingWait {
    MTL::SharedEvent* event;
    uint64_t value;
    const char* kind;
  };
  std::vector<PendingWait> pending_waits_;
  // Waits of the most recently committed command buffer (diagnostics).
  std::vector<PendingWait> last_committed_waits_;
  // Kernels encoded into the open / last committed command buffer (for the
  // timeout diagnostics and the reset log).
  std::vector<std::shared_ptr<const KernelIdentity>> pending_kernels_;
  std::shared_ptr<const std::vector<std::shared_ptr<const KernelIdentity>>>
      last_committed_kernels_;
  uint64_t last_committed_signal_ = 0;
 public:
  // Diagnostics: one line describing this stream's fence state and what its
  // last committed command buffer waited on.
  std::string DebugState();
 private:
  // First GPU failure (set by a failed command buffer's completion handler or
  // by Synchronize); sticky. `async_error_reported_` is set once Synchronize
  // has returned it.
  std::mutex err_mu_;
  absl::Status async_error_;
  bool async_error_reported_ = false;
  // FailedPreconditionError if the stream is in the error state.
  absl::Status CheckAsyncError();

  // Host work ordered on the stream: each task waits for the fence to reach
  // wait_value, runs, then signals signal_value so later GPU work proceeds.
  struct HostTask {
    uint64_t wait_value;
    std::function<void()> fn;
    uint64_t signal_value;
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
