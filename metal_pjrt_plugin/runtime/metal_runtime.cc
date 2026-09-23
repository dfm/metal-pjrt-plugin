#include "metal_pjrt_plugin/runtime/metal_runtime.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstring>
#include <thread>
#include <functional>
#include <sstream>

namespace metal_pjrt {
namespace rt {

namespace {

std::string NSErrorToString(NS::Error* err) {
  if (err == nullptr) return "unknown Metal error";
  return err->localizedDescription()->utf8String();
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
  std::ostringstream os;
  os << std::hex << h << "_" << s.size();
  return os.str();
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

Status Device::Create(int ordinal, std::unique_ptr<Device>* out) {
  NS::Array* devices = MTL::CopyAllDevices();
  if (devices == nullptr || ordinal < 0 ||
      ordinal >= static_cast<int>(devices->count())) {
    if (devices) devices->release();
    return Status("Metal device ordinal out of range: " +
                  std::to_string(ordinal));
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
  *out = std::move(dev);
  return Status::Ok();
}

Device::~Device() {
  for (auto& kv : pso_cache_) kv.second->release();
  for (auto& kv : library_cache_) kv.second->release();
  for (auto& kv : allocations_) kv.second.first->release();
  if (device_) device_->release();
}

Status Device::Allocate(uint64_t size, Allocation* out) {
  if (size == 0) size = 1;
  if (size > info_.max_buffer_length) {
    return Status("allocation of " + std::to_string(size) +
                  " bytes exceeds Metal maxBufferLength " +
                  std::to_string(info_.max_buffer_length));
  }
  // Default (tracked) hazard mode: Metal orders dispatches touching the same
  // buffer for us. Untracked mode with manual barriers is a later optimization.
  MTL::Buffer* buf = device_->newBuffer(
      size, MTL::ResourceStorageModeShared);
  if (buf == nullptr) {
    return Status("Metal newBuffer failed for " + std::to_string(size) +
                  " bytes");
  }
  void* ptr = buf->contents();
  {
    std::lock_guard<std::mutex> lock(mu_);
    allocations_[reinterpret_cast<uintptr_t>(ptr)] = {buf, size};
    allocated_bytes_ += size;
  }
  out->ptr = ptr;
  out->size = size;
  return Status::Ok();
}

Status Device::Deallocate(void* ptr) {
  if (ptr == nullptr) return Status::Ok();
  MTL::Buffer* buf = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == allocations_.end()) {
      return Status("Deallocate: pointer was not allocated by this device");
    }
    buf = it->second.first;
    allocated_bytes_ -= it->second.second;
    allocations_.erase(it);
  }
  buf->release();
  return Status::Ok();
}

Status Device::Resolve(const void* ptr, BufferRef* out) const {
  std::lock_guard<std::mutex> lock(mu_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = allocations_.upper_bound(addr);
  if (it == allocations_.begin()) {
    return Status("Resolve: pointer not in any device allocation");
  }
  --it;
  uintptr_t base = it->first;
  uint64_t size = it->second.second;
  if (addr < base || addr >= base + size) {
    return Status("Resolve: pointer not in any device allocation");
  }
  out->buffer = it->second.first;
  out->offset = addr - base;
  return Status::Ok();
}

uint64_t Device::allocated_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return allocated_bytes_;
}

Status Device::CompileLibrary(const std::string& msl_source,
                              MTL::Library** out) {
  std::string key = HashString(msl_source);
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = library_cache_.find(key);
    if (it != library_cache_.end()) {
      *out = it->second;
      return Status::Ok();
    }
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  NS::Error* err = nullptr;
  MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
  opts->setFastMathEnabled(false);
  MTL::Library* lib = device_->newLibrary(Str(msl_source), opts, &err);
  opts->release();
  Status status;
  if (lib == nullptr) {
    status = Status("Metal shader compilation failed: " + NSErrorToString(err));
  } else {
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = library_cache_.emplace(key, lib);
    if (!inserted) {
      lib->release();  // lost a race; use the cached one
    }
    *out = it->second;
  }
  pool->release();
  return status;
}

Status Device::CreateKernel(MTL::Library* library, const std::string& function,
                            std::unique_ptr<Kernel>* out) {
  std::string key =
      std::to_string(reinterpret_cast<uintptr_t>(library)) + ":" + function;
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
      return Status("Metal function not found in library: " + function);
    }
    NS::Error* err = nullptr;
    pso = device_->newComputePipelineState(fn, &err);
    fn->release();
    if (pso == nullptr) {
      std::string msg = NSErrorToString(err);
      pool->release();
      return Status("Metal pipeline creation failed for " + function + ": " +
                    msg);
    }
    pool->release();
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = pso_cache_.emplace(key, pso);
    if (!inserted) {
      pso->release();
      pso = it->second;
    }
  }
  out->reset(new Kernel(pso, function));
  return Status::Ok();
}

Status Device::CreateStream(std::unique_ptr<Stream>* out) {
  MTL::CommandQueue* q = device_->newCommandQueue();
  if (q == nullptr) return Status("Metal newCommandQueue failed");
  MTL::SharedEvent* fence = device_->newSharedEvent();
  if (fence == nullptr) {
    q->release();
    return Status("Metal newSharedEvent failed");
  }
  out->reset(new Stream(this, q, fence));
  return Status::Ok();
}

Status Device::CreateEvent(std::unique_ptr<Event>* out) {
  MTL::SharedEvent* ev = device_->newSharedEvent();
  if (ev == nullptr) return Status("Metal newSharedEvent failed");
  out->reset(new Event(ev));
  return Status::Ok();
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

Status Event::WaitOnHost() {
  uint64_t v;
  {
    std::lock_guard<std::mutex> lock(mu_);
    v = value_;
  }
  if (v == 0) return Status::Ok();
  while (!event_->waitUntilSignaledValue(v, /*milliseconds=*/1000)) {
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Stream

Stream::Stream(Device* device, MTL::CommandQueue* queue, MTL::SharedEvent* fence)
    : device_(device), queue_(queue), fence_(fence) {
  worker_ = std::thread([this]() { WorkerLoop(); });
}

Stream::~Stream() {
  Synchronize();
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

Status Stream::CheckAsyncError() {
  std::lock_guard<std::mutex> lock(err_mu_);
  if (async_error_.empty()) return Status::Ok();
  return Status("Metal stream is in error state: " + async_error_);
}

Status Stream::EnsureCommandBuffer() {
  if (cmd_ != nullptr) return Status::Ok();
  Status err = CheckAsyncError();
  if (!err.ok()) return err;
  // Retained references: XLA frees device buffers as soon as the host no
  // longer needs them and relies on the driver to keep memory alive until
  // enqueued GPU work completes (as CUDA does). Metal only does that for
  // retained command buffers; unretained ones fault with
  // kIOGPUCommandBufferCallbackErrorInvalidResource.
  cmd_ = queue_->commandBuffer();
  if (cmd_ == nullptr) return Status("Metal commandBuffer creation failed");
  cmd_->retain();
  ops_in_cmd_ = 0;
  return Status::Ok();
}

Status Stream::EnsureComputeEncoder() {
  Status s = EnsureCommandBuffer();
  if (!s.ok()) return s;
  if (enc_ == nullptr) {
    enc_ = cmd_->computeCommandEncoder(MTL::DispatchTypeSerial);
    if (enc_ == nullptr) return Status("Metal computeCommandEncoder failed");
    enc_->retain();
  }
  return Status::Ok();
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

Status Stream::Commit() {
  if (cmd_ == nullptr) return Status::Ok();
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
  cmd_->addCompletedHandler([this, v, fence, signals](MTL::CommandBuffer* cb) {
    if (cb->status() == MTL::CommandBufferStatusError) {
      std::string msg = NSErrorToString(cb->error());
      fprintf(stderr, "[metal-pjrt] GPU command buffer failed: %s\n",
              msg.c_str());
      {
        std::lock_guard<std::mutex> lock(err_mu_);
        if (async_error_.empty()) async_error_ = msg;
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
  return Status::Ok();
}

Status Stream::Flush() {
  std::lock_guard<std::mutex> lock(mu_);
  return Commit();
}

Status Stream::Synchronize() {
  std::lock_guard<std::mutex> lock(mu_);
  Status s = Commit();
  if (!s.ok()) return s;
  // Wait for completion (not just the fence signal) so resources referenced
  // by these command buffers may be freed by the caller right away.
  for (MTL::CommandBuffer* cb : in_flight_) {
    cb->waitUntilCompleted();
    if (cb->status() == MTL::CommandBufferStatusError) {
      NS::Error* err = cb->error();
      last_error_ = "Metal command buffer failed: " + NSErrorToString(err);
    }
    cb->release();
  }
  in_flight_.clear();
  {
    Status err = CheckAsyncError();
    if (!err.ok()) return err;
  }
  if (!last_error_.empty()) {
    std::string e;
    e.swap(last_error_);
    return Status(e);
  }
  return Status::Ok();
}

Status Stream::Launch(const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
                      const std::vector<KernelArg>& args,
                      uint32_t threadgroup_memory_bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  Status s = EnsureComputeEncoder();
  if (!s.ok()) return s;
  enc_->setComputePipelineState(kernel.pso());
  for (size_t i = 0; i < args.size(); ++i) {
    const KernelArg& a = args[i];
    if (a.is_buffer) {
      BufferRef ref;
      s = device_->Resolve(a.device_ptr, &ref);
      if (!s.ok()) {
        return Status("Launch " + kernel.name() + " arg " + std::to_string(i) +
                      ": " + s.message());
      }
      enc_->setBuffer(ref.buffer, ref.offset, static_cast<NS::UInteger>(i));
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
  return Status::Ok();
}

Status Stream::EncodeExternal(
    std::function<Status(void* mtl_command_buffer)> encode) {
  std::lock_guard<std::mutex> lock(mu_);
  Status s = EnsureCommandBuffer();
  if (!s.ok()) return s;
  EndEncoder();
  s = encode(static_cast<void*>(cmd_));
  if (!s.ok()) return s;
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return Status::Ok();
}

Status Stream::MemcpyDeviceToDevice(void* dst, const void* src, uint64_t size) {
  if (size == 0) return Status::Ok();
  std::lock_guard<std::mutex> lock(mu_);
  BufferRef d, s;
  Status st = device_->Resolve(dst, &d);
  if (!st.ok()) return st;
  st = device_->Resolve(src, &s);
  if (!st.ok()) return st;
  st = EnsureCommandBuffer();
  if (!st.ok()) return st;
  EndEncoder();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  blit->copyFromBuffer(s.buffer, s.offset, d.buffer, d.offset, size);
  blit->endEncoding();
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return Status::Ok();
}

Status Stream::Memset8(void* dst, uint8_t value, uint64_t size) {
  if (size == 0) return Status::Ok();
  std::lock_guard<std::mutex> lock(mu_);
  BufferRef d;
  Status st = device_->Resolve(dst, &d);
  if (!st.ok()) return st;
  st = EnsureCommandBuffer();
  if (!st.ok()) return st;
  EndEncoder();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  blit->fillBuffer(d.buffer, NS::Range(d.offset, size), value);
  blit->endEncoding();
  if (++ops_in_cmd_ >= kMaxOpsPerCommandBuffer) return Commit();
  return Status::Ok();
}

Status Stream::Memset32(void* dst, uint32_t value, uint64_t size) {
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

Status Stream::HostCallback(std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(mu_);
  // 1. Commit everything so far; it ends with a fence signal (value v).
  Status s = Commit();
  if (!s.ok()) return s;
  // 2. Open a new command buffer whose first action is to wait on a value
  //    the host callback will signal after running.
  s = EnsureCommandBuffer();
  if (!s.ok()) return s;
  uint64_t done_value = ++fence_value_;
  uint64_t after_prior = last_committed_fence_value_;
  {
    std::lock_guard<std::mutex> lock(work_mu_);
    work_.push_back(HostTask{after_prior, std::move(fn), done_value});
  }
  work_cv_.notify_one();
  cmd_->encodeWait(fence_, done_value);
  return Status::Ok();
}

Status Stream::MemcpyHostToDevice(void* dst, const void* src, uint64_t size) {
  if (size == 0) return Status::Ok();
  return HostCallback([dst, src, size]() { std::memcpy(dst, src, size); });
}

Status Stream::MemcpyDeviceToHost(void* dst, const void* src, uint64_t size) {
  if (size == 0) return Status::Ok();
  return HostCallback([dst, src, size]() { std::memcpy(dst, src, size); });
}

Status Stream::RecordEvent(Event* event) {
  std::lock_guard<std::mutex> lock(mu_);
  Status s = EnsureCommandBuffer();
  if (!s.ok()) return s;
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

Status Stream::WaitForEvent(Event* event) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t v;
  {
    std::lock_guard<std::mutex> elock(event->mu_);
    v = event->value_;
  }
  if (v == 0) return Status::Ok();
  Status s = EnsureCommandBuffer();
  if (!s.ok()) return s;
  EndEncoder();
  cmd_->encodeWait(event->event_, v);
  return Status::Ok();
}

Status Stream::WaitForStream(Stream* other) {
  if (other == this) return Status::Ok();
  uint64_t v;
  {
    std::lock_guard<std::mutex> olock(other->mu_);
    Status s = other->Commit();
    if (!s.ok()) return s;
    v = other->last_committed_fence_value_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  Status s = EnsureCommandBuffer();
  if (!s.ok()) return s;
  EndEncoder();
  cmd_->encodeWait(other->fence_, v);
  return Status::Ok();
}

}  // namespace rt
}  // namespace metal_pjrt
