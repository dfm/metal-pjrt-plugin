// Thin C++ runtime over Metal (via metal-cpp), independent of XLA so it can be
// unit-tested standalone. The StreamExecutor adapter translates this API into
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
#ifndef METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_
#define METAL_PJRT_PLUGIN_RUNTIME_METAL_RUNTIME_H_

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

// Minimal status type so this layer stays free of absl.
class Status {
 public:
  Status() = default;
  explicit Status(std::string message) : message_(std::move(message)) {}
  static Status Ok() { return Status(); }
  bool ok() const { return message_.empty(); }
  const std::string& message() const { return message_; }

 private:
  std::string message_;
};

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
  std::vector<uint8_t> bytes;
};

class Kernel {
 public:
  Kernel(MTL::ComputePipelineState* pso, std::string name)
      : pso_(pso), name_(std::move(name)) {}
  ~Kernel();
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  MTL::ComputePipelineState* pso() const { return pso_; }
  const std::string& name() const { return name_; }
  uint32_t max_total_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;
  uint32_t static_threadgroup_memory_length() const;

 private:
  MTL::ComputePipelineState* pso_;
  std::string name_;
};

class Event;
class Stream;

class Device {
 public:
  static Status Create(int ordinal, std::unique_ptr<Device>* out);
  static int VisibleDeviceCount();
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceInfo& info() const { return info_; }
  int ordinal() const { return ordinal_; }
  MTL::Device* mtl() const { return device_; }

  // Memory. Allocations are shared-storage MTL::Buffers; freeing an address
  // that was not returned by Allocate is an error.
  Status Allocate(uint64_t size, Allocation* out);
  Status Deallocate(void* ptr);
  // Resolve a raw pointer (possibly interior) to its buffer and offset.
  Status Resolve(const void* ptr, BufferRef* out) const;
  uint64_t allocated_bytes() const;

  // Compile Metal Shading Language source into a library, cached by content.
  Status CompileLibrary(const std::string& msl_source, MTL::Library** out);
  // Create (or fetch cached) pipeline state for `function` in `library`.
  Status CreateKernel(MTL::Library* library, const std::string& function,
                      std::unique_ptr<Kernel>* out);

  Status CreateStream(std::unique_ptr<Stream>* out);
  Status CreateEvent(std::unique_ptr<Event>* out);

 private:
  Device() = default;
  int ordinal_ = 0;
  MTL::Device* device_ = nullptr;
  DeviceInfo info_;

  mutable std::mutex mu_;
  // Keyed by start address; value is the buffer and its size.
  std::map<uintptr_t, std::pair<MTL::Buffer*, uint64_t>> allocations_;
  uint64_t allocated_bytes_ = 0;
  std::unordered_map<std::string, MTL::Library*> library_cache_;  // key: source hash
  std::unordered_map<std::string, MTL::ComputePipelineState*> pso_cache_;
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
  Status WaitOnHost();

 private:
  friend class Device;
  friend class Stream;
  explicit Event(MTL::SharedEvent* ev) : event_(ev) {}
  MTL::SharedEvent* event_;
  uint64_t value_ = 0;  // last recorded value; 0 means never recorded
  std::mutex mu_;
};

class Stream {
 public:
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  // Kernel launch. threadgroups = grid in threadgroup units, threads = per
  // threadgroup. Buffer args are resolved to (buffer, offset) here.
  Status Launch(const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
                const std::vector<KernelArg>& args,
                uint32_t threadgroup_memory_bytes = 0);

  // Encode work produced outside this runtime (e.g. Metal Performance
  // Shaders) into the stream's open command buffer, in order with all other
  // stream work. Ends the current compute encoder, then calls `encode` with the
  // raw MTL::CommandBuffer* (as void*, so Objective-C++ callers can bridge it
  // to id<MTLCommandBuffer>). `encode` must create, use and end its own
  // encoders and must not commit the buffer or call back into this stream (the
  // stream lock is held). Counts as one op toward kMaxOpsPerCommandBuffer.
  Status EncodeExternal(std::function<Status(void* mtl_command_buffer)> encode);

  // Device-to-device copy and fill, via a blit encoder.
  Status MemcpyDeviceToDevice(void* dst, const void* src, uint64_t size);
  Status Memset8(void* dst, uint8_t value, uint64_t size);
  Status Memset32(void* dst, uint32_t value, uint64_t size);

  // Host transfers are ordered on the stream: they run in a host callback once
  // prior work completes, and later stream work waits for them. With unified
  // memory these are plain memcpys.
  Status MemcpyHostToDevice(void* dst, const void* src, uint64_t size);
  Status MemcpyDeviceToHost(void* dst, const void* src, uint64_t size);

  // Run `fn` on the host once all previously enqueued work has completed;
  // subsequent stream work waits for it to finish.
  Status HostCallback(std::function<void()> fn);

  Status RecordEvent(Event* event);
  Status WaitForEvent(Event* event);
  // Make this stream wait for everything currently enqueued on `other`.
  Status WaitForStream(Stream* other);

  // Commit any open work and block until the stream is idle.
  Status Synchronize();
  // Commit any open work without waiting.
  Status Flush();

  // Number of dispatches encoded into the current open command buffer before
  // it is automatically committed.
  static constexpr int kMaxOpsPerCommandBuffer = 64;

 private:
  friend class Device;
  Stream(Device* device, MTL::CommandQueue* queue, MTL::SharedEvent* fence);

  // Ensure an open command buffer/encoder exist.
  Status EnsureCommandBuffer();
  Status EnsureComputeEncoder();
  void EndEncoder();
  // Commit the current command buffer (if any).
  Status Commit();
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
  // Committed but possibly still executing command buffers (retained).
  // Synchronize waits for their completion, not just the fence signal, so
  // callers may free resources immediately afterwards.
  std::vector<MTL::CommandBuffer*> in_flight_;
  std::string last_error_;
  std::mutex mu_;
  // Events signaled by the open command buffer; on GPU error they are
  // force-signaled so host waiters wake up instead of hanging.
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> pending_signals_;
  // Set by the completion handler of a failed command buffer; sticky.
  std::mutex err_mu_;
  std::string async_error_;
  Status CheckAsyncError();

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
