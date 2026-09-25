#include "metal_pjrt_plugin/runtime/metal_runtime.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace metal_pjrt {
namespace rt {

namespace {

// METAL_PJRT_TRACE=1 logs one line per committed command buffer with its op
// count and GPU execution time. Diagnostic only.
bool TraceEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("METAL_PJRT_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

// "<description> (domain=<domain>, code=<code>)".
std::string NSErrorToString(NS::Error* err) {
  if (err == nullptr) return "unknown Metal error (no NSError)";
  NS::String* desc = err->localizedDescription();
  NS::String* domain = err->domain();
  return absl::StrCat(desc ? desc->utf8String() : "(no description)",
                      " (domain=", domain ? domain->utf8String() : "?",
                      ", code=", static_cast<int64_t>(err->code()), ")");
}

NS::String* Str(const std::string& s) {
  return NS::String::string(s.c_str(), NS::UTF8StringEncoding);
}

// FNV-1a; good enough as a cache key for MSL sources.
std::string HashString(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return absl::StrCat(absl::Hex(h), "_", s.size());
}

int HighestAppleFamily(MTL::Device* d) {
  static const MTL::GPUFamily families[] = {
      MTL::GPUFamilyApple9, MTL::GPUFamilyApple8, MTL::GPUFamilyApple7,
      MTL::GPUFamilyApple6, MTL::GPUFamilyApple5, MTL::GPUFamilyApple4};
  static const int numbers[] = {9, 8, 7, 6, 5, 4};
  for (int i = 0; i < 6; ++i) {
    if (d->supportsFamily(families[i])) return numbers[i];
  }
  return 0;
}

// Prefixes `context` to a non-OK status, keeping its code.
absl::Status Annotate(const absl::Status& s, absl::string_view context) {
  if (s.ok()) return s;
  return absl::Status(s.code(), absl::StrCat(context, ": ", s.message()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel

Kernel::~Kernel() {
  // Pipeline states are owned by the Device cache; nothing to release here.
}

uint32_t Kernel::max_total_threads_per_threadgroup() const {
  return static_cast<uint32_t>(pso_->maxTotalThreadsPerThreadgroup());
}

uint32_t Kernel::thread_execution_width() const {
  return static_cast<uint32_t>(pso_->threadExecutionWidth());
}

uint32_t Kernel::static_threadgroup_memory_length() const {
  return static_cast<uint32_t>(pso_->staticThreadgroupMemoryLength());
}

// ---------------------------------------------------------------------------
// Device

int Device::VisibleDeviceCount() {
  NS::Array* devices = MTL::CopyAllDevices();
  int n = devices ? static_cast<int>(devices->count()) : 0;
  if (devices) devices->release();
  return n;
}

absl::StatusOr<std::unique_ptr<Device>> Device::Create(int ordinal) {
  NS::Array* devices = MTL::CopyAllDevices();
  const int count = devices ? static_cast<int>(devices->count()) : 0;
  if (ordinal < 0 || ordinal >= count) {
    if (devices) devices->release();
    return absl::InvalidArgumentError(absl::StrFormat(
        "Metal device ordinal %d out of range; %d device(s) visible", ordinal,
        count));
  }
  MTL::Device* d = devices->object<MTL::Device>(ordinal);
  d->retain();
  devices->release();

  std::unique_ptr<Device> dev(new Device());
  dev->ordinal_ = ordinal;
  dev->device_ = d;
  DeviceInfo& info = dev->info_;
  info.name = d->name()->utf8String();
  info.unified_memory = d->hasUnifiedMemory();
  info.max_buffer_length = d->maxBufferLength();
  info.recommended_working_set = d->recommendedMaxWorkingSetSize();
  info.max_threads_per_threadgroup =
      static_cast<uint32_t>(d->maxThreadsPerThreadgroup().width);
  info.threadgroup_memory_length =
      static_cast<uint32_t>(d->maxThreadgroupMemoryLength());
  info.gpu_family = HighestAppleFamily(d);
  info.supports_metal4 = info.gpu_family >= 9;
  info.simd_width = 32;  // All Apple GPUs; confirmed per-pipeline on creation.
  return dev;
}

Device::~Device() {
  if (!allocations_.empty()) {
    LOG(ERROR) << "Metal device " << ordinal_ << " destroyed with "
               << allocations_.size() << " live allocation(s) totalling "
               << allocated_bytes_ << " bytes; releasing them";
  }
  for (auto& kv : pso_cache_) kv.second->release();
  for (auto& kv : library_cache_) kv.second->release();
  for (auto& kv : allocations_) kv.second.first->release();
  if (device_) device_->release();
}

absl::StatusOr<Allocation> Device::Allocate(uint64_t size) {
  const uint64_t requested = size;
  if (size == 0) size = 1;
  if (size > info_.max_buffer_length) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal allocation of %d bytes exceeds the device's maxBufferLength of "
        "%d bytes (device %d, %s)",
        requested, info_.max_buffer_length, ordinal_, info_.name));
  }
  // Default (tracked) hazard mode: Metal orders dispatches touching the same
  // buffer for us. Untracked mode with manual barriers is a later optimization.
  MTL::Buffer* buf = device_->newBuffer(size, MTL::ResourceStorageModeShared);
  if (buf == nullptr) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal newBuffer failed for %d bytes on device %d (%s): %d bytes "
        "already allocated, recommended working set %d bytes, maxBufferLength "
        "%d bytes",
        requested, ordinal_, info_.name, allocated_bytes(),
        info_.recommended_working_set, info_.max_buffer_length));
  }
  void* ptr = buf->contents();
  {
    std::lock_guard<std::mutex> lock(mu_);
    allocations_[reinterpret_cast<uintptr_t>(ptr)] = {buf, size};
    allocated_bytes_ += size;
  }
  return Allocation{ptr, size};
}

absl::Status Device::Deallocate(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  MTL::Buffer* buf = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == allocations_.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Metal Deallocate: %p is not the start of an allocation on device %d",
          ptr, ordinal_));
    }
    buf = it->second.first;
    allocated_bytes_ -= it->second.second;
    allocations_.erase(it);
  }
  buf->release();
  return absl::OkStatus();
}

absl::StatusOr<BufferRef> Device::Resolve(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mu_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = allocations_.upper_bound(addr);
  if (it != allocations_.begin()) {
    --it;
    uintptr_t base = it->first;
    uint64_t size = it->second.second;
    if (addr >= base && addr < base + size) {
      return BufferRef{it->second.first, addr - base};
    }
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "pointer %p is not inside any allocation on Metal device %d", ptr,
      ordinal_));
}

uint64_t Device::allocated_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return allocated_bytes_;
}

absl::StatusOr<MTL::Library*> Device::CompileLibrary(
    const std::string& msl_source) {
  std::string key = HashString(msl_source);
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = library_cache_.find(key);
    if (it != library_cache_.end()) return it->second;
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  NS::Error* err = nullptr;
  MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
  opts->setFastMathEnabled(false);
  MTL::Library* lib = device_->newLibrary(Str(msl_source), opts, &err);
  opts->release();
  absl::StatusOr<MTL::Library*> result;
  if (lib == nullptr) {
    // The error lives in the autorelease pool; format it before draining.
    result = absl::InternalError(absl::StrFormat(
        "Metal shader compilation failed (%d bytes of MSL, key %s): %s",
        msl_source.size(), key, NSErrorToString(err)));
  } else {
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = library_cache_.emplace(key, lib);
    if (!inserted) {
      lib->release();  // lost a race; use the cached one
    }
    result = it->second;
  }
  pool->release();
  return result;
}

absl::StatusOr<std::unique_ptr<Kernel>> Device::CreateKernel(
    MTL::Library* library, const std::string& function) {
  if (library == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CreateKernel(", function, "): null library"));
  }
  std::string key =
      absl::StrCat(reinterpret_cast<uintptr_t>(library), ":", function);
  MTL::ComputePipelineState* pso = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = pso_cache_.find(key);
    if (it != pso_cache_.end()) pso = it->second;
  }
  if (pso == nullptr) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function* fn = library->newFunction(Str(function));
    if (fn == nullptr) {
      pool->release();
      return absl::InvalidArgumentError(absl::StrCat(
          "Metal function '", function, "' not found in library"));
    }
    NS::Error* err = nullptr;
    pso = device_->newComputePipelineState(fn, &err);
    fn->release();
    if (pso == nullptr) {
      std::string msg = NSErrorToString(err);
      pool->release();
      return absl::InternalError(absl::StrCat(
          "Metal compute pipeline creation failed for '", function, "': ",
          msg));
    }
    pool->release();
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = pso_cache_.emplace(key, pso);
    if (!inserted) {
      pso->release();
      pso = it->second;
    }
  }
  return std::make_unique<Kernel>(pso, function);
}

absl::StatusOr<std::unique_ptr<Stream>> Device::CreateStream() {
  MTL::CommandQueue* q = device_->newCommandQueue();
  if (q == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Metal newCommandQueue failed on device %d (%s)", ordinal_,
        info_.name));
  }
  MTL::SharedEvent* fence = device_->newSharedEvent();
  if (fence == nullptr) {
    q->release();
    return absl::InternalError(absl::StrFormat(
        "Metal newSharedEvent failed creating a stream fence on device %d",
        ordinal_));
  }
  return std::unique_ptr<Stream>(new Stream(this, q, fence));
}

absl::StatusOr<std::unique_ptr<Event>> Device::CreateEvent() {
  MTL::SharedEvent* ev = device_->newSharedEvent();
  if (ev == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Metal newSharedEvent failed creating an event on device %d",
        ordinal_));
  }
  return std::unique_ptr<Event>(new Event(ev));
}

// ---------------------------------------------------------------------------
// Event

Event::~Event() { event_->release(); }

bool Event::IsComplete() const {
  uint64_t v;
  {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mu_));
    v = value_;
  }
  return event_->signaledValue() >= v;
}

absl::Status Event::WaitOnHost() {
  uint64_t v;
  {
    std::lock_guard<std::mutex> lock(mu_);
    v = value_;
  }
  if (v == 0) return absl::OkStatus();
  // A failed command buffer force-signals its events (see Stream::Commit), so
  // this cannot hang on GPU errors; the error is reported by the stream.
  while (!event_->waitUntilSignaledValue(v, /*milliseconds=*/1000)) {
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Stream

Stream::Stream(Device* device, MTL::CommandQueue* queue, MTL::SharedEvent* fence)
    : device_(device), queue_(queue), fence_(fence) {
  worker_ = std::thread([this]() { WorkerLoop(); });
}

Stream::~Stream() {
  absl::Status s = Synchronize();
  if (!s.ok()) {
    LOG(ERROR) << "Metal stream on device " << device_->ordinal()
               << " had pending errors at destruction: " << s;
  }
  {
    std::lock_guard<std::mutex> lock(work_mu_);
    stop_ = true;
  }
  work_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  if (enc_) {
    enc_->endEncoding();
    enc_->release();
  }
  if (cmd_) cmd_->release();
  for (MTL::CommandBuffer* cb : in_flight_) cb->release();
  fence_->release();
  queue_->release();
}

void Stream::WorkerLoop() {
  while (true) {
    HostTask task;
    {
      std::unique_lock<std::mutex> lock(work_mu_);
      work_cv_.wait(lock, [this]() { return stop_ || !work_.empty(); });
      if (work_.empty()) return;  // stop_ and drained
      task = std::move(work_.front());
      work_.pop_front();
    }
    while (!fence_->waitUntilSignaledValue(task.wait_value, 1000)) {
    }
    task.fn();
    fence_->setSignaledValue(task.signal_value);
  }
}

absl::Status Stream::CheckAsyncError() {
  std::lock_guard<std::mutex> lock(err_mu_);
  if (async_error_.ok()) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat(
      "Metal stream on device ", device_->ordinal(),
      " is in error state after an earlier GPU failure: ",
      async_error_.message()));
}

absl::Status Stream::EnsureCommandBuffer() {
  if (cmd_ != nullptr) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(CheckAsyncError());
  // Retained references: XLA frees device buffers as soon as the host no
  // longer needs them and relies on the driver to keep memory alive until
  // enqueued GPU work completes (as CUDA does). Metal only does that for
  // retained command buffers; unretained ones fault with
  // kIOGPUCommandBufferCallbackErrorInvalidResource.
  cmd_ = queue_->commandBuffer();
  if (cmd_ == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Metal commandBuffer creation failed on device ", device_->ordinal()));
  }
  cmd_->retain();
  ops_in_cmd_ = 0;
  return absl::OkStatus();
}

absl::Status Stream::EnsureComputeEncoder() {
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  if (enc_ == nullptr) {
    enc_ = cmd_->computeCommandEncoder(MTL::DispatchTypeSerial);
    if (enc_ == nullptr) {
      return absl::InternalError("Metal computeCommandEncoder creation failed");
    }
    enc_->retain();
  }
  return absl::OkStatus();
}

void Stream::EndEncoder() {
  if (enc_ != nullptr) {
    enc_->endEncoding();
    enc_->release();
    enc_ = nullptr;
  }
}

uint64_t Stream::SignalFence() {
  // Caller holds mu_ and has an open command buffer with no open encoder.
  ++fence_value_;
  cmd_->encodeSignalEvent(fence_, fence_value_);
  return fence_value_;
}

absl::Status Stream::Commit() {
  if (cmd_ == nullptr) return absl::OkStatus();
  EndEncoder();
  // Every command buffer ends by signaling the stream timeline, so
  // Synchronize/WaitForStream have a value to wait on.
  uint64_t v = SignalFence();
  // On failure, log, record the error and force-signal the fence and any
  // events this buffer was supposed to signal, so waiters do not hang.
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> signals;
  signals.swap(pending_signals_);
  for (auto& sv : signals) sv.first->retain();
  MTL::SharedEvent* fence = fence_;
  fence->retain();
  const int ordinal = device_->ordinal();
  const int traced_ops = ops_in_cmd_;
  cmd_->addCompletedHandler(
      [this, v, fence, signals, ordinal, traced_ops](MTL::CommandBuffer* cb) {
        if (TraceEnabled()) {
          LOG(ERROR) << "[metal-trace] stream " << this << " cb#" << v
                    << " ops=" << traced_ops << " gpu_ms="
                    << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        }
        if (cb->status() == MTL::CommandBufferStatusError) {
          absl::Status error = absl::InternalError(absl::StrFormat(
              "Metal command buffer failed on device %d (stream fence value "
              "%d): %s",
              ordinal, v, NSErrorToString(cb->error())));
          LOG(ERROR) << error.message();
          {
            std::lock_guard<std::mutex> lock(err_mu_);
            if (async_error_.ok()) async_error_ = error;
          }
          if (fence->signaledValue() < v) fence->setSignaledValue(v);
          for (auto& sv : signals) {
            if (sv.first->signaledValue() < sv.second) {
              sv.first->setSignaledValue(sv.second);
            }
          }
        }
        for (auto& sv : signals) sv.first->release();
        fence->release();
      });
  cmd_->commit();
  // Keep the (retained) buffer until it completes; prune finished ones.
  in_flight_.push_back(cmd_);
  cmd_ = nullptr;
  ops_in_cmd_ = 0;
  last_committed_fence_value_ = v;
  std::vector<MTL::CommandBuffer*> still_running;
  for (MTL::CommandBuffer* cb : in_flight_) {
    MTL::CommandBufferStatus st = cb->status();
    if (st == MTL::CommandBufferStatusCompleted ||
        st == MTL::CommandBufferStatusError) {
      cb->release();
    } else {
      still_running.push_back(cb);
    }
  }
  in_flight_.swap(still_running);
  return absl::OkStatus();
}

absl::Status Stream::Flush() {
  std::lock_guard<std::mutex> lock(mu_);
  return Commit();
}

absl::Status Stream::Synchronize() {
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(Commit());
  // Wait for completion (not just the fence signal) so resources referenced
  // by these command buffers may be freed by the caller right away.
  absl::Status first_failure;
  for (MTL::CommandBuffer* cb : in_flight_) {
    cb->waitUntilCompleted();
    if (cb->status() == MTL::CommandBufferStatusError && first_failure.ok()) {
      first_failure = absl::InternalError(
          absl::StrCat("Metal command buffer failed on device ",
                       device_->ordinal(), ": ", NSErrorToString(cb->error())));
    }
    cb->release();
  }
  in_flight_.clear();
  // The completion handler may not have run yet; record the failure here too
  // so the error state does not depend on that ordering.
  std::lock_guard<std::mutex> elock(err_mu_);
  if (async_error_.ok()) async_error_ = first_failure;
  if (async_error_.ok()) return absl::OkStatus();
  if (!async_error_reported_) {
    async_error_reported_ = true;
    return async_error_;
  }
  return absl::FailedPreconditionError(absl::StrCat(
      "Metal stream on device ", device_->ordinal(),
      " is in error state after an earlier GPU failure: ",
      async_error_.message()));
}

bool UsesArgumentBuffer(const std::string& msl_source,
                        const std::string& kernel_name) {
  return msl_source.find(absl::StrCat(kArgumentBufferMarker, "\nkernel void ",
                                      kernel_name, "(")) != std::string::npos;
}

absl::Status Stream::Launch(const Kernel& kernel, Dim3 threadgroups,
                            Dim3 threads, const std::vector<KernelArg>& args,
                            uint32_t threadgroup_memory_bytes) {
  size_t num_buffers = 0;
  for (const KernelArg& a : args) num_buffers += a.is_buffer ? 1 : 0;
  if (kernel.uses_argument_buffer()) {
    return LaunchWithArgumentBuffer(kernel, threadgroups, threads, args,
                                    threadgroup_memory_bytes);
  }
  if (args.size() > kMaxBufferArgs) {
    return absl::UnimplementedError(absl::StrFormat(
        "Launch %s: %d arguments (%d buffers) exceed Metal's %d argument "
        "table slots; argument buffers are not implemented",
        kernel.name(), args.size(), num_buffers, kMaxBufferArgs));
  }
  std::lock_guard<std::mutex> lock(mu_);
  // Resolve before opening an encoder so a bad pointer encodes nothing.
  std::vector<BufferRef> refs(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].is_buffer) continue;
    absl::StatusOr<BufferRef> ref = device_->Resolve(args[i].device_ptr);
    if (!ref.ok()) {
      return Annotate(ref.status(), absl::StrFormat("Launch %s argument %d",
                                                    kernel.name(), i));
    }
    refs[i] = *ref;
  }
  ABSL_RETURN_IF_ERROR(EnsureComputeEncoder());
  enc_->setComputePipelineState(kernel.pso());
  for (size_t i = 0; i < args.size(); ++i) {
    const KernelArg& a = args[i];
    if (a.is_buffer) {
      enc_->setBuffer(refs[i].buffer, refs[i].offset,
                      static_cast<NS::UInteger>(i));
    } else {
      enc_->setBytes(a.bytes.data(), a.bytes.size(),
                     static_cast<NS::UInteger>(i));
    }
  }
  if (threadgroup_memory_bytes > 0) {
    enc_->setThreadgroupMemoryLength(threadgroup_memory_bytes, 0);
  }
  enc_->dispatchThreadgroups(
      MTL::Size(threadgroups.x, threadgroups.y, threadgroups.z),
      MTL::Size(threads.x, threads.y, threads.z));
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return absl::OkStatus();
}

absl::Status Stream::LaunchWithArgumentBuffer(
    const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
    const std::vector<KernelArg>& args, uint32_t threadgroup_memory_bytes) {
  if (args.size() > kMaxArgumentBufferArgs) {
    return absl::UnimplementedError(absl::StrFormat(
        "Launch %s: %d buffer arguments exceed the argument-buffer limit of %d",
        kernel.name(), args.size(), kMaxArgumentBufferArgs));
  }
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<uint64_t> addrs(std::max<size_t>(args.size(), 1), 0);
  std::vector<MTL::Buffer*> resident;
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].is_buffer) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Launch %s argument %d: kernels with an argument buffer take only "
          "buffer arguments",
          kernel.name(), i));
    }
    absl::StatusOr<BufferRef> ref = device_->Resolve(args[i].device_ptr);
    if (!ref.ok()) {
      return Annotate(ref.status(), absl::StrFormat("Launch %s argument %d",
                                                    kernel.name(), i));
    }
    addrs[i] = ref->buffer->gpuAddress() + ref->offset;
    if (std::find(resident.begin(), resident.end(), ref->buffer) ==
        resident.end()) {
      resident.push_back(ref->buffer);
    }
  }
  ABSL_RETURN_IF_ERROR(EnsureComputeEncoder());
  enc_->setComputePipelineState(kernel.pso());
  enc_->setBytes(addrs.data(), addrs.size() * sizeof(uint64_t), 0);
  // Buffers reached only through GPU addresses must be made resident (and
  // visible to hazard tracking) explicitly.
  if (!resident.empty()) {
    enc_->useResources(
        reinterpret_cast<const MTL::Resource* const*>(resident.data()),
        resident.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (threadgroup_memory_bytes > 0) {
    enc_->setThreadgroupMemoryLength(threadgroup_memory_bytes, 0);
  }
  enc_->dispatchThreadgroups(
      MTL::Size(threadgroups.x, threadgroups.y, threadgroups.z),
      MTL::Size(threads.x, threads.y, threads.z));
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return absl::OkStatus();
}

absl::Status Stream::EncodeExternal(
    std::function<absl::Status(void* mtl_command_buffer)> encode) {
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  ABSL_RETURN_IF_ERROR(encode(static_cast<void*>(cmd_)));
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return absl::OkStatus();
}

absl::Status Stream::MemcpyDeviceToDevice(void* dst, const void* src,
                                          uint64_t size) {
  if (size == 0) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mu_);
  const std::string what =
      absl::StrFormat("MemcpyDeviceToDevice(%p <- %p, %d bytes)", dst, src, size);
  absl::StatusOr<BufferRef> d = device_->Resolve(dst);
  if (!d.ok()) return Annotate(d.status(), absl::StrCat(what, " destination"));
  absl::StatusOr<BufferRef> s = device_->Resolve(src);
  if (!s.ok()) return Annotate(s.status(), absl::StrCat(what, " source"));
  if (d->offset + size > d->buffer->length() ||
      s->offset + size > s->buffer->length()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s runs past the end of an allocation (dst %d/%d, src %d/%d bytes)",
        what, d->offset, d->buffer->length(), s->offset, s->buffer->length()));
  }
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  if (blit == nullptr) {
    return absl::InternalError(
        absl::StrCat("Metal blitCommandEncoder creation failed for ", what));
  }
  blit->copyFromBuffer(s->buffer, s->offset, d->buffer, d->offset, size);
  blit->endEncoding();
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return absl::OkStatus();
}

absl::Status Stream::Memset8(void* dst, uint8_t value, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mu_);
  const std::string what =
      absl::StrFormat("Memset8(%p, 0x%02x, %d bytes)", dst, value, size);
  absl::StatusOr<BufferRef> d = device_->Resolve(dst);
  if (!d.ok()) return Annotate(d.status(), what);
  if (d->offset + size > d->buffer->length()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s runs past the end of its %d-byte allocation (offset %d)", what,
        d->buffer->length(), d->offset));
  }
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  if (blit == nullptr) {
    return absl::InternalError(
        absl::StrCat("Metal blitCommandEncoder creation failed for ", what));
  }
  blit->fillBuffer(d->buffer, NS::Range(d->offset, size), value);
  blit->endEncoding();
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return absl::OkStatus();
}

absl::Status Stream::Memset32(void* dst, uint32_t value, uint64_t size) {
  if (size % 4 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Memset32(%p, 0x%08x, %d bytes): size must be a multiple of 4", dst,
        value, size));
  }
  // Blit fill is byte-granular; a 32-bit pattern needs a kernel or a host
  // callback. Use the host path for correctness; a kernel can replace it.
  uint8_t b = static_cast<uint8_t>(value & 0xff);
  if ((value >> 8 & 0xff) == b && (value >> 16 & 0xff) == b &&
      (value >> 24 & 0xff) == b) {
    return Memset8(dst, b, size);
  }
  return HostCallback([dst, value, size]() {
    uint32_t* p = static_cast<uint32_t*>(dst);
    for (uint64_t i = 0; i < size / 4; ++i) p[i] = value;
  });
}

absl::Status Stream::HostCallback(std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(mu_);
  // 1. Commit everything so far; it ends with a fence signal (value v).
  ABSL_RETURN_IF_ERROR(Commit());
  // 2. Open a new command buffer whose first action is to wait on a value
  //    the host callback will signal after running.
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  uint64_t done_value = ++fence_value_;
  uint64_t after_prior = last_committed_fence_value_;
  {
    std::lock_guard<std::mutex> lock(work_mu_);
    work_.push_back(HostTask{after_prior, std::move(fn), done_value});
  }
  work_cv_.notify_one();
  cmd_->encodeWait(fence_, done_value);
  return absl::OkStatus();
}

absl::Status Stream::MemcpyHostToDevice(void* dst, const void* src,
                                        uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MemcpyHostToDevice(%p <- %p, %d bytes): null pointer", dst, src,
        size));
  }
  return HostCallback([dst, src, size]() { std::memcpy(dst, src, size); });
}

absl::Status Stream::MemcpyDeviceToHost(void* dst, const void* src,
                                        uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MemcpyDeviceToHost(%p <- %p, %d bytes): null pointer", dst, src,
        size));
  }
  return HostCallback([dst, src, size]() { std::memcpy(dst, src, size); });
}

absl::Status Stream::RecordEvent(Event* event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("RecordEvent: null event");
  }
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  uint64_t v;
  {
    std::lock_guard<std::mutex> elock(event->mu_);
    v = ++event->value_;
  }
  cmd_->encodeSignalEvent(event->event_, v);
  pending_signals_.emplace_back(event->event_, v);
  // Commit so a host or another stream waiting on the event can make progress.
  return Commit();
}

absl::Status Stream::WaitForEvent(Event* event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("WaitForEvent: null event");
  }
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t v;
  {
    std::lock_guard<std::mutex> elock(event->mu_);
    v = event->value_;
  }
  if (v == 0) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  cmd_->encodeWait(event->event_, v);
  return absl::OkStatus();
}

absl::Status Stream::WaitForStream(Stream* other) {
  if (other == nullptr) {
    return absl::InvalidArgumentError("WaitForStream: null stream");
  }
  if (other == this) return absl::OkStatus();
  uint64_t v;
  {
    std::lock_guard<std::mutex> olock(other->mu_);
    ABSL_RETURN_IF_ERROR(other->Commit());
    v = other->last_committed_fence_value_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  cmd_->encodeWait(other->fence_, v);
  return absl::OkStatus();
}

}  // namespace rt
}  // namespace metal_pjrt
