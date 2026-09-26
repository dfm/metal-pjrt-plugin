#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/runtime/system_memory.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <mach/mach.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <limits>
#include <functional>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
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

int MaxOpsPerCommandBuffer() {
  static const int n = [] {
    const char* v = std::getenv("METAL_PJRT_MAX_OPS");
    int x = v ? std::atoi(v) : 0;
    return x > 0 ? x : Stream::kMaxOpsPerCommandBuffer;
  }();
  return n;
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
// Waits on the host for `ev` to reach `v`, giving up once the device is lost
// (work committed after a reset may never run, and an unbounded wait leaves
// the process unable to exit, which is how the driver got wedged once).
absl::Status WaitForValueOnHost(Device* device, MTL::SharedEvent* ev,
                                uint64_t v) {
  while (!ev->waitUntilSignaledValue(v, /*milliseconds=*/200)) {
    ABSL_RETURN_IF_ERROR(device->lost_status());
  }
  return absl::OkStatus();
}

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
  info.simd_width = 32;
  {
    // The pool may grow to 3/4 of physical RAM (XLA's own default for
    // dedicated GPUs), capped by the GPU's recommended working set. This is a
    // ceiling, not a reservation: regions are only mapped when used, and the
    // allocation-time guard below refuses growth that would leave the system
    // without free memory, so a busy machine degrades to clean
    // RESOURCE_EXHAUSTED errors instead of swapping.
    // Half of RAM: freed pool chunks stay resident (BFC only releases regions
    // when growth is refused), and a GPU stalling on swapped-out pages is what
    // trips the watchdog. Override with JAX_METAL_MEMORY_FRACTION (of this).
    uint64_t budget = PhysicalMemoryBytes() / 2;
    budget = std::min(budget, static_cast<uint64_t>(info.recommended_working_set));
    dev->memory_budget_ = budget;
    LOG(INFO) << "Metal device " << ordinal << " (" << info.name
              << "): memory budget " << (budget >> 20) << " MB, "
              << (ReclaimableMemoryBytes() >> 20) << " MB reclaimable now";
  }  // All Apple GPUs; confirmed per-pipeline on creation.
  dev->LoadResetLog();
  return dev;
}

namespace {
std::string ResetLogPath(const std::string& state_dir) {
  return absl::StrCat(state_dir, "/gpu_resets.jsonl");
}

int64_t BootTimeSeconds() {
  struct timeval tv;
  size_t len = sizeof(tv);
  if (sysctlbyname("kern.boottime", &tv, &len, nullptr, 0) != 0) return 0;
  return tv.tv_sec;
}

// Minimal JSON string escaping for the messages we write.
std::string JsonEscape(absl::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char ch : s) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          absl::StrAppendFormat(&out, "\\u%04x", ch);
        } else {
          out += ch;
        }
    }
  }
  return out;
}
}  // namespace

void Device::LoadResetLog() {
  const char* dir = std::getenv("METAL_PJRT_STATE_DIR");
  if (dir != nullptr && dir[0] != '\0') {
    state_dir_ = dir;
  } else {
    const char* home = std::getenv("HOME");
    state_dir_ = absl::StrCat(home != nullptr ? home : ".", "/.cache/jax_metal");
  }
  if (const char* v = std::getenv("METAL_PJRT_QUARANTINE_STRIKES")) {
    int n = 0;
    if (absl::SimpleAtoi(v, &n) && n >= 0) {
      quarantine_strikes_ = n;
    } else {
      LOG(WARNING) << "Ignoring METAL_PJRT_QUARANTINE_STRIKES=" << v;
    }
  }
  std::ifstream in(ResetLogPath(state_dir_));
  if (!in) return;
  const int64_t boot = BootTimeSeconds();
  std::string line;
  std::lock_guard<std::mutex> lock(mu_);
  while (std::getline(in, line)) {
    // Each line is one reset: {"time":N,...,"kernels":[{"key":"..",..},..]}.
    // The keys are hex hashes plus a function name, so a substring scan is
    // enough; no JSON parser in the runtime.
    size_t t = line.find("\"time\":");
    int64_t when = 0;
    if (t != std::string::npos) {
      absl::SimpleAtoi(absl::string_view(line).substr(t + 7,
                                                   line.find_first_of(",}", t) - (t + 7)),
                       &when);
    }
    if (when >= boot) ++resets_since_boot_;
    size_t pos = 0;
    while ((pos = line.find("\"key\":\"", pos)) != std::string::npos) {
      pos += 7;
      size_t end = line.find('"', pos);
      if (end == std::string::npos) break;
      ++strikes_[line.substr(pos, end - pos)];
      pos = end;
    }
  }
  if (resets_since_boot_ > 0) {
    LOG(WARNING) << "The GPU was reset " << resets_since_boot_
                 << " time(s) since boot (" << ResetLogPath(state_dir_)
                 << "); the driver leaves it slow after resets, so timings "
                    "are unreliable until a reboot";
  }
}

void Device::RecordReset(
    absl::string_view cause,
    absl::Span<const std::shared_ptr<const KernelIdentity>> kernels) {
  const int64_t now = static_cast<int64_t>(std::time(nullptr));
  std::string line = absl::StrFormat(
      "{\"time\":%d,\"pid\":%d,\"device\":%d,\"error\":\"%s\",\"kernels\":[",
      now, static_cast<int>(getpid()), ordinal_, JsonEscape(cause));
  // Each kernel once, in launch order.
  std::vector<const KernelIdentity*> unique;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& k : kernels) {
      if (k == nullptr) continue;
      auto [it, inserted] = strikes_.emplace(k->key, 0);
      if (std::find_if(unique.begin(), unique.end(), [&](const KernelIdentity* u) {
            return u->key == k->key;
          }) == unique.end()) {
        unique.push_back(k.get());
        ++it->second;
      }
    }
  }
  for (size_t i = 0; i < unique.size(); ++i) {
    absl::StrAppendFormat(&line, "%s{\"key\":\"%s\",\"name\":\"%s\"}",
                          i ? "," : "", JsonEscape(unique[i]->key),
                          JsonEscape(unique[i]->name));
  }
  line += "]}\n";
  std::error_code ec;
  std::filesystem::create_directories(state_dir_, ec);
  std::ofstream out(ResetLogPath(state_dir_), std::ios::app);
  if (!out) {
    LOG(ERROR) << "Could not append the GPU reset record to "
               << ResetLogPath(state_dir_);
    return;
  }
  out << line;
  LOG(ERROR) << "Recorded the GPU reset in " << ResetLogPath(state_dir_)
             << " with " << unique.size()
             << " suspect kernel(s); kernels seen in " << quarantine_strikes_
             << " resets are refused until the log is cleared "
                "(scripts/gpu_health.py --clear)";
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
  // Refuse allocations that would push the system into swap (that is how the
  // machine got wedged once).
  uint64_t reclaimable = 0;
  if (!FitsInSystemMemory(size, &reclaimable)) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal allocation of %d bytes refused: only %d bytes of system "
        "memory are reclaimable and %d bytes are reserved for the OS "
        "(device %d, %d bytes already allocated by this process, budget %d "
        "bytes). Reduce the working set or close other memory-heavy "
        "processes.",
        requested, reclaimable, kSystemMemoryReserve, ordinal_,
        allocated_bytes(), memory_budget_));
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
  // METAL_PJRT_POISON_ALLOCATIONS=1 fills fresh device memory with 0xFF (NaN
  // for floats, huge for ints) so reads of uninitialized memory are visible
  // and deterministic instead of depending on what a recycled buffer held.
  static const bool poison = [] {
    const char* v = std::getenv("METAL_PJRT_POISON_ALLOCATIONS");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (poison) std::memset(ptr, 0xFF, size);
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
    allocation_generation_.fetch_add(1, std::memory_order_acq_rel);
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

absl::Status Device::lost_status() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lost_;
}

void Device::MarkLost(const absl::Status& why) {
  std::lock_guard<std::mutex> lock(mu_);
  if (lost_.ok()) {
    lost_ = absl::FailedPreconditionError(absl::StrCat(
        "the Metal GPU was reset while this process had work in flight; no "
        "further GPU work is accepted in this process (restart it). Cause: ",
        why.message()));
  }
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
    library_keys_[it->second] = key;
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
  auto identity = std::make_shared<KernelIdentity>();
  identity->name = function;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto lk = library_keys_.find(library);
    identity->key = absl::StrCat(
        lk == library_keys_.end() ? "unknown" : lk->second, ":", function);
    auto strikes = strikes_.find(identity->key);
    if (quarantine_strikes_ > 0 && strikes != strikes_.end() &&
        strikes->second >= quarantine_strikes_) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "kernel %s is quarantined: it was in the command buffer that timed "
          "out in %d GPU watchdog resets (see %s/gpu_resets.jsonl). Fix the "
          "kernel, then clear the log with scripts/gpu_health.py --clear; "
          "METAL_PJRT_QUARANTINE_STRIKES=0 disables the quarantine",
          function, strikes->second, state_dir_));
    }
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
  return std::make_unique<Kernel>(pso, std::move(identity));
}

namespace {
// Built-in kernels. Fill: `v` is the pattern broadcast to 32 bits, `n` the
// number of elements (words or bytes) to write; byte i takes byte (i mod 4) of
// the pattern, which is right for 1-, 2- and 4-byte patterns alike. Copy:
// 16-byte vectors when both ends are 16-byte aligned, bytes otherwise.
constexpr char kBuiltinMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
kernel void xla_metal_fill32(device uint* p [[buffer(0)]],
                             constant uint& v [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < n) p[i] = v;
}
kernel void xla_metal_fill8(device uchar* p [[buffer(0)]],
                            constant uint& v [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < n) p[i] = (uchar)((v >> (8u * (i & 3u))) & 0xffu);
}
kernel void xla_metal_copy16(device const uint4* src [[buffer(0)]],
                             device uint4* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
kernel void xla_metal_copy8(device const uchar* src [[buffer(0)]],
                            device uchar* dst [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
)";
constexpr const char* kBuiltinNames[] = {"xla_metal_fill32", "xla_metal_fill8",
                                         "xla_metal_copy16", "xla_metal_copy8"};
}  // namespace

absl::StatusOr<const Kernel*> Device::BuiltinKernel(Builtin kind) {
  std::lock_guard<std::mutex> lock(builtin_mu_);
  std::unique_ptr<Kernel>& slot = builtin_kernels_[static_cast<int>(kind)];
  if (slot == nullptr) {
    ABSL_ASSIGN_OR_RETURN(MTL::Library * lib, CompileLibrary(kBuiltinMsl));
    ABSL_ASSIGN_OR_RETURN(slot,
                          CreateKernel(lib, kBuiltinNames[static_cast<int>(kind)]));
  }
  return slot.get();
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
  return std::unique_ptr<Event>(new Event(this, ev));
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
  return WaitForValueOnHost(device_, event_, v);
}

// ---------------------------------------------------------------------------
// Stream

Stream::Stream(Device* device, MTL::CommandQueue* queue, MTL::SharedEvent* fence)
    : device_(device), queue_(queue), fence_(fence) {
  device_->RegisterStream(this);
  worker_ = std::thread([this]() { WorkerLoop(); });
}

Stream::~Stream() {
  device_->UnregisterStream(this);
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
  for (const PendingWait& w : last_committed_waits_) w.event->release();
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
    bool abandoned = !WaitForValueOnHost(device_, fence_, task.wait_value).ok();
    for (auto& [ev, value] : task.extra_waits) {
      if (!abandoned) abandoned = !WaitForValueOnHost(device_, ev, value).ok();
      ev->release();
    }
    if (abandoned) {
      // Unblock any GPU wait on this task without running the callback.
      if (fence_->signaledValue() < task.signal_value) {
        fence_->setSignaledValue(task.signal_value);
      }
      continue;
    }
    task.fn();
    // A failed command buffer may have force-signaled the fence past this
    // value meanwhile; never move it backwards.
    if (fence_->signaledValue() < task.signal_value) {
      fence_->setSignaledValue(task.signal_value);
    }
  }
}

absl::Status Stream::CheckAsyncError() {
  std::lock_guard<std::mutex> lock(completion_->mu);
  if (completion_->error.ok()) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat(
      "Metal stream on device ", device_->ordinal(),
      " is in error state after an earlier GPU failure: ",
      completion_->error.message()));
}

absl::Status Stream::EnsureCommandBuffer() {
  if (cmd_ != nullptr) {
    FlushDeferredWaits();
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(device_->lost_status());
  // Retained references: XLA frees device buffers as soon as the host no
  // longer needs them and relies on the driver to keep memory alive until
  // enqueued GPU work completes (as CUDA does). Metal only does that for
  // retained command buffers; unretained ones fault with
  // kIOGPUCommandBufferCallbackErrorInvalidResource.
  // Callers are XLA threads without an autorelease pool; without one the
  // autoreleased command buffer would stay referenced until thread exit.
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  cmd_ = queue_->commandBuffer();
  if (cmd_ != nullptr) cmd_->retain();
  pool->release();
  if (cmd_ == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Metal commandBuffer creation failed on device ", device_->ordinal()));
  }
  ops_in_cmd_ = 0;
  threads_in_cmd_ = 0;
  FlushDeferredWaits();
  return absl::OkStatus();
}

void Stream::FlushDeferredWaits() {
  if (deferred_waits_.empty()) return;
  EndEncoder();
  for (const PendingWait& w : deferred_waits_) {
    cmd_->encodeWait(w.event, w.value);
    pending_waits_.push_back(w);
  }
  deferred_waits_.clear();
}

absl::Status Stream::EnsureComputeEncoder() {
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  if (enc_ == nullptr) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    enc_ = cmd_->computeCommandEncoder(MTL::DispatchTypeSerial);
    if (enc_ != nullptr) enc_->retain();
    pool->release();
    if (enc_ == nullptr) {
      return absl::InternalError("Metal computeCommandEncoder creation failed");
    }
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
  std::vector<PendingWait> waits;
  waits.swap(pending_waits_);
  auto kernels =
      std::make_shared<const std::vector<std::shared_ptr<const KernelIdentity>>>(
          std::move(pending_kernels_));
  pending_kernels_.clear();
  // The diagnostic copy of the waits holds its own references: DebugState
  // reads the events' signaled values later, possibly from another stream's
  // completion handler after the waited-on event or stream is gone.
  std::vector<PendingWait> previous_waits = waits;
  for (auto& w : previous_waits) w.event->retain();
  {
    std::lock_guard<std::mutex> lock(diag_mu_);
    last_committed_waits_.swap(previous_waits);
    last_committed_kernels_ = kernels;
  }
  for (auto& w : previous_waits) w.event->release();
  last_committed_signal_ = v;
  for (auto& w : waits) w.event->retain();
  MTL::SharedEvent* fence = fence_;
  fence->retain();
  const int ordinal = device_->ordinal();
  const int traced_ops = ops_in_cmd_;
  // The handler must not touch `this`: after a device reset ~Stream stops
  // waiting for in-flight buffers, whose handlers may run later.
  Device* device = device_;
  std::shared_ptr<CompletionState> state = completion_;
  const void* self = this;
  state->pending.fetch_add(1, std::memory_order_relaxed);
  cmd_->addCompletedHandler(
      [device, state, self, v, fence, signals, waits, ordinal, traced_ops,
       kernels](MTL::CommandBuffer* cb) {
        if (TraceEnabled()) {
          LOG(ERROR) << "[metal-trace] stream " << self << " cb#" << v
                    << " ops=" << traced_ops << " gpu_ms="
                    << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        }
        if (cb->status() == MTL::CommandBufferStatusError) {
          absl::Status error = absl::InternalError(absl::StrFormat(
              "Metal command buffer failed on device %d (stream fence value "
              "%d): %s",
              ordinal, v, NSErrorToString(cb->error())));
          LOG(ERROR) << error.message();
          // Diagnostics: a timeout on a buffer with little GPU work usually
          // means it sat on a wait whose signal never came. Say which.
          for (const PendingWait& w : waits) {
            LOG(ERROR) << "  this command buffer waited on " << w.kind
                       << " event " << w.event << " for value " << w.value
                       << "; its signaled value is now "
                       << w.event->signaledValue();
          }
          LOG(ERROR) << "  ops in buffer: " << traced_ops << ", gpu time ms: "
                     << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
          device->DumpStreams();
          {
            std::lock_guard<std::mutex> lock(state->mu);
            if (state->error.ok()) state->error = error;
          }
          // Watchdog timeouts and revoked/removed devices mean the GPU was
          // reset underneath us: stop submitting from this process.
          const long code = cb->error() ? cb->error()->code() : 0;
          if (code == MTL::CommandBufferErrorTimeout ||
              code == MTL::CommandBufferErrorAccessRevoked ||
              code == MTL::CommandBufferErrorDeviceRemoved) {
            device->RecordReset(error.message(), *kernels);
            device->MarkLost(error);
          }
          if (fence->signaledValue() < v) fence->setSignaledValue(v);
          for (auto& sv : signals) {
            if (sv.first->signaledValue() < sv.second) {
              sv.first->setSignaledValue(sv.second);
            }
          }
        }
        for (auto& sv : signals) sv.first->release();
        for (auto& w : waits) w.event->release();
        fence->release();
        state->pending.fetch_sub(1, std::memory_order_release);
      });
  cmd_->commit();
  last_commit_time_ = std::chrono::steady_clock::now();
  // Keep the (retained) buffer until it completes; prune finished ones.
  in_flight_.push_back(cmd_);
  cmd_ = nullptr;
  ops_in_cmd_ = 0;
  threads_in_cmd_ = 0;
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
    // Bounded wait: after a device reset, committed work may never run.
    while (true) {
      MTL::CommandBufferStatus st = cb->status();
      if (st == MTL::CommandBufferStatusCompleted ||
          st == MTL::CommandBufferStatusError) {
        break;
      }
      if (!device_->lost_status().ok()) break;  // abandon it
      if (fence_->signaledValue() >= last_committed_fence_value_) {
        // The buffer's last action (the fence signal) has run, so its status
        // flips to Completed as soon as the completion handler fires; waiting
        // for that cannot hang. (A timed sleep here cost ~1 ms per
        // Synchronize and dominated sync-heavy programs.)
        cb->waitUntilCompleted();
        break;
      }
      fence_->waitUntilSignaledValue(last_committed_fence_value_, 200);
    }
    if (cb->status() == MTL::CommandBufferStatusError && first_failure.ok()) {
      first_failure = absl::InternalError(
          absl::StrCat("Metal command buffer failed on device ",
                       device_->ordinal(), ": ", NSErrorToString(cb->error())));
    }
    cb->release();
  }
  in_flight_.clear();
  // Waits not yet encoded are satisfied on the host instead of by a
  // wait-only command buffer (which would count against the GPU watchdog).
  for (const PendingWait& w : deferred_waits_) {
    absl::Status s = WaitForValueOnHost(device_, w.event, w.value);
    if (!s.ok() && first_failure.ok()) first_failure = s;
  }
  if (first_failure.ok()) deferred_waits_.clear();
  // Work abandoned above because the device was reset must not look done.
  if (first_failure.ok()) first_failure = device_->lost_status();
  // The completion handler may not have run yet; record the failure here too
  // so the error state does not depend on that ordering.
  std::lock_guard<std::mutex> elock(completion_->mu);
  absl::Status& async_error = completion_->error;
  if (async_error.ok()) async_error = first_failure;
  if (async_error.ok()) return absl::OkStatus();
  // A failed command buffer does not take the device down (unlike a CUDA
  // context error), so report the failure once, to the caller waiting on
  // this work, and let the stream recover for subsequent work.
  absl::Status error = async_error;
  absl::Status lost = device_->lost_status();
  if (!lost.ok()) return lost;  // sticky: the GPU was reset
  async_error = absl::OkStatus();
  return error;
}

bool UsesArgumentBuffer(const std::string& msl_source,
                        const std::string& kernel_name) {
  return msl_source.find(absl::StrCat(kArgumentBufferMarker, "\nkernel void ",
                                      kernel_name, "(")) != std::string::npos;
}

absl::StatusOr<BufferRef> Stream::ResolveCached(const void* ptr) {
  const uint64_t gen = device_->allocation_generation();
  if (gen != resolve_cache_generation_) {
    for (CachedRange& c : resolve_cache_) c = CachedRange();
    resolve_cache_generation_ = gen;
  }
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  for (const CachedRange& c : resolve_cache_) {
    if (addr - c.base < c.size) return BufferRef{c.buffer, addr - c.base};
  }
  absl::StatusOr<BufferRef> ref = device_->Resolve(ptr);
  if (ref.ok()) {
    CachedRange& c = resolve_cache_[resolve_cache_next_];
    resolve_cache_next_ = (resolve_cache_next_ + 1) % kResolveCacheSize;
    c.base = addr - ref->offset;
    c.size = ref->buffer->length();
    c.buffer = ref->buffer;
  }
  return ref;
}

namespace {
std::chrono::microseconds EarlyCommitInterval() {
  static const std::chrono::microseconds v = [] {
    const char* e = std::getenv("METAL_PJRT_EARLY_COMMIT_US");
    int x = e ? std::atoi(e) : -1;
    return std::chrono::microseconds(x >= 0 ? x
                                            : Stream::kEarlyCommitIntervalUs);
  }();
  return v;
}
}  // namespace

absl::Status Stream::FinishOp(uint64_t work) {
  threads_in_cmd_ += work;
  ++ops_in_cmd_;
  if (ops_in_cmd_ >= MaxOpsPerCommandBuffer() ||
      threads_in_cmd_ >= kMaxThreadsPerCommandBuffer) {
    return Commit();
  }
  if (ops_in_cmd_ >= kEarlyCommitOps &&
      completion_->pending.load(std::memory_order_acquire) == 0 &&
      std::chrono::steady_clock::now() - last_commit_time_ >=
          EarlyCommitInterval()) {
    return Commit();
  }
  return absl::OkStatus();
}

namespace {
// Launch argument checks.
absl::Status ValidateLaunch(const Kernel& kernel, Dim3 threads,
                            absl::Span<const KernelArg> args) {
  if (kernel.uses_argument_buffer()) {
    if (args.size() > Stream::kMaxArgumentBufferArgs) {
      return absl::UnimplementedError(absl::StrFormat(
          "Launch %s: %d buffer arguments exceed the argument-buffer limit "
          "of %d",
          kernel.name(), args.size(), Stream::kMaxArgumentBufferArgs));
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (!args[i].is_buffer) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Launch %s argument %d: kernels with an argument buffer take only "
            "buffer arguments",
            kernel.name(), i));
      }
    }
  } else if (args.size() > Stream::kMaxBufferArgs) {
    size_t num_buffers = 0;
    for (const KernelArg& a : args) num_buffers += a.is_buffer ? 1 : 0;
    return absl::UnimplementedError(absl::StrFormat(
        "Launch %s: %d arguments (%d buffers) exceed Metal's %d argument "
        "table slots and the kernel has no argument buffer",
        kernel.name(), args.size(), num_buffers, Stream::kMaxBufferArgs));
  }
  const uint64_t per_group =
      static_cast<uint64_t>(threads.x) * threads.y * threads.z;
  if (per_group == 0 || per_group > kernel.max_total_threads_per_threadgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Launch %s: %d threads per threadgroup (pipeline limit %d)",
        kernel.name(), per_group, kernel.max_total_threads_per_threadgroup()));
  }
  return absl::OkStatus();
}

uint64_t LaunchWork(Dim3 threadgroups, Dim3 threads) {
  return static_cast<uint64_t>(threadgroups.x) * threadgroups.y *
         threadgroups.z * threads.x * threads.y * threads.z;
}
}  // namespace

absl::Status Stream::Launch(const Kernel& kernel, Dim3 threadgroups,
                            Dim3 threads, absl::Span<const KernelArg> args,
                            uint32_t threadgroup_memory_bytes) {
  ABSL_RETURN_IF_ERROR(ValidateLaunch(kernel, threads, args));
  if (kernel.uses_argument_buffer()) {
    return LaunchWithArgumentBuffer(kernel, threadgroups, threads, args,
                                    threadgroup_memory_bytes);
  }
  std::lock_guard<std::mutex> lock(mu_);
  // Resolve before opening an encoder so a bad pointer encodes nothing.
  BufferRef refs[kMaxBufferArgs];
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].is_buffer) continue;
    absl::StatusOr<BufferRef> ref = ResolveCached(args[i].device_ptr);
    if (!ref.ok()) {
      return Annotate(ref.status(), absl::StrFormat("Launch %s argument %d",
                                                    kernel.name(), i));
    }
    refs[i] = *ref;
  }
  return EncodeLaunch(kernel, threadgroups, threads, args,
                      absl::MakeConstSpan(refs, args.size()),
                      threadgroup_memory_bytes);
}

absl::Status Stream::LaunchWithArgumentBuffer(
    const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
    absl::Span<const KernelArg> args, uint32_t threadgroup_memory_bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  absl::InlinedVector<uint64_t, 64> addrs(std::max<size_t>(args.size(), 1), 0);
  absl::InlinedVector<MTL::Buffer*, 16> resident;
  for (size_t i = 0; i < args.size(); ++i) {
    absl::StatusOr<BufferRef> ref = ResolveCached(args[i].device_ptr);
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
  return EncodeLaunchWithArgumentBuffer(kernel, threadgroups, threads, addrs,
                                        resident, threadgroup_memory_bytes);
}

absl::Status Stream::EncodeLaunch(const Kernel& kernel, Dim3 threadgroups,
                                  Dim3 threads,
                                  absl::Span<const KernelArg> args,
                                  absl::Span<const BufferRef> refs,
                                  uint32_t threadgroup_memory_bytes) {
  ABSL_RETURN_IF_ERROR(EnsureComputeEncoder());
  enc_->setComputePipelineState(kernel.pso());
  pending_kernels_.push_back(kernel.identity());
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
  return FinishOp(LaunchWork(threadgroups, threads));
}

absl::Status Stream::EncodeLaunchWithArgumentBuffer(
    const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
    absl::Span<const uint64_t> addrs, absl::Span<MTL::Buffer* const> resident,
    uint32_t threadgroup_memory_bytes) {
  ABSL_RETURN_IF_ERROR(EnsureComputeEncoder());
  enc_->setComputePipelineState(kernel.pso());
  pending_kernels_.push_back(kernel.identity());
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
  return FinishOp(LaunchWork(threadgroups, threads));
}

absl::Status Stream::EncodeExternal(
    std::function<absl::Status(void* mtl_command_buffer)> encode) {
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  ABSL_RETURN_IF_ERROR(encode(static_cast<void*>(cmd_)));
  return FinishOp(kExternalOpWork);
}

namespace {
absl::Status CheckRange(const BufferRef& ref, uint64_t size,
                        absl::string_view what) {
  if (ref.offset + size > ref.buffer->length()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s runs past the end of its %d-byte allocation (offset %d, %d "
        "bytes)",
        what, ref.buffer->length(), ref.offset, size));
  }
  return absl::OkStatus();
}
}  // namespace

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
  ABSL_RETURN_IF_ERROR(CheckRange(*d, size, absl::StrCat(what, " destination")));
  ABSL_RETURN_IF_ERROR(CheckRange(*s, size, absl::StrCat(what, " source")));
  return EncodeCopy(*d, *s, size);
}

// One-dimensional dispatch of a built-in kernel over n elements: buffers
// 0..k-1 then the element count at index k. Caller holds mu_.
absl::Status Stream::EncodeBuiltin(Device::Builtin kind,
                                   absl::Span<const BufferRef> buffers,
                                   const void* bytes, size_t bytes_len,
                                   uint64_t n64) {
  if (n64 > std::numeric_limits<uint32_t>::max()) {
    return absl::UnimplementedError(absl::StrFormat(
        "built-in kernel over %d elements exceeds the 32-bit limit", n64));
  }
  ABSL_ASSIGN_OR_RETURN(const Kernel* kernel, device_->BuiltinKernel(kind));
  const uint32_t n = static_cast<uint32_t>(n64);
  ABSL_RETURN_IF_ERROR(EnsureComputeEncoder());
  enc_->setComputePipelineState(kernel->pso());
  pending_kernels_.push_back(kernel->identity());
  NS::UInteger slot = 0;
  for (const BufferRef& b : buffers) enc_->setBuffer(b.buffer, b.offset, slot++);
  if (bytes != nullptr) enc_->setBytes(bytes, bytes_len, slot++);
  enc_->setBytes(&n, sizeof(n), slot);
  const uint32_t group =
      std::min<uint32_t>(n, kernel->max_total_threads_per_threadgroup());
  enc_->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(group, 1, 1));
  return FinishOp(n);
}

absl::Status Stream::EncodeCopy(BufferRef dst, BufferRef src, uint64_t size) {
  if (size <= kComputeCopyMaxBytes) {
    const bool vec = size % 16 == 0 && dst.offset % 16 == 0 && src.offset % 16 == 0;
    return EncodeBuiltin(vec ? Device::Builtin::kCopy16 : Device::Builtin::kCopy8,
                         {src, dst}, nullptr, 0, vec ? size / 16 : size);
  }
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  if (blit == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Metal blitCommandEncoder creation failed for a %d-byte copy", size));
  }
  blit->copyFromBuffer(src.buffer, src.offset, dst.buffer, dst.offset, size);
  blit->endEncoding();
  return FinishOp(size / 4);  // about one thread per word
}

absl::Status Stream::Memset8(void* dst, uint8_t value, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mu_);
  const std::string what =
      absl::StrFormat("Memset8(%p, 0x%02x, %d bytes)", dst, value, size);
  absl::StatusOr<BufferRef> d = device_->Resolve(dst);
  if (!d.ok()) return Annotate(d.status(), what);
  ABSL_RETURN_IF_ERROR(CheckRange(*d, size, what));
  const uint32_t v = value;
  return EncodeFill(*d, v | v << 8 | v << 16 | v << 24, 1, size);
}

absl::Status Stream::Memset32(void* dst, uint32_t value, uint64_t size) {
  if (size % 4 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Memset32(%p, 0x%08x, %d bytes): size must be a multiple of 4", dst,
        value, size));
  }
  if (size == 0) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mu_);
  const std::string what =
      absl::StrFormat("Memset32(%p, 0x%08x, %d bytes)", dst, value, size);
  absl::StatusOr<BufferRef> d = device_->Resolve(dst);
  if (!d.ok()) return Annotate(d.status(), what);
  ABSL_RETURN_IF_ERROR(CheckRange(*d, size, what));
  return EncodeFill(*d, value, 4, size);
}

absl::Status Stream::EncodeFill(BufferRef dst, uint32_t pattern,
                                int pattern_bytes, uint64_t size) {
  const uint8_t b = static_cast<uint8_t>(pattern & 0xff);
  const bool uniform = (pattern >> 8 & 0xff) == b &&
                       (pattern >> 16 & 0xff) == b && (pattern >> 24 & 0xff) == b;
  if (uniform && size > kComputeCopyMaxBytes) {
    ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
    EndEncoder();
    MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
    if (blit == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Metal blitCommandEncoder creation failed for a %d-byte fill", size));
    }
    blit->fillBuffer(dst.buffer, NS::Range(dst.offset, size), b);
    blit->endEncoding();
    return FinishOp(size / 4);
  }
  // Word-granular when the range is 4-byte aligned; bytes otherwise.
  const bool words = size % 4 == 0 && dst.offset % 4 == 0;
  return EncodeBuiltin(words ? Device::Builtin::kFill32 : Device::Builtin::kFill8,
                       {dst}, &pattern, sizeof(pattern), words ? size / 4 : size);
}

absl::Status Stream::HostCallback(std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(mu_);
  // 1. Commit everything so far; it ends with a fence signal (value v). Waits
  //    still deferred on this stream become waits of the host task itself.
  ABSL_RETURN_IF_ERROR(Commit());
  ABSL_RETURN_IF_ERROR(device_->lost_status());
  uint64_t done_value = ++fence_value_;
  uint64_t after_prior = last_committed_fence_value_;
  HostTask task{after_prior, std::move(fn), done_value, {}};
  for (const PendingWait& w : deferred_waits_) {
    w.event->retain();
    task.extra_waits.emplace_back(w.event, w.value);
  }
  deferred_waits_.clear();
  {
    std::lock_guard<std::mutex> lock(work_mu_);
    work_.push_back(std::move(task));
  }
  work_cv_.notify_one();
  // 2. Later GPU work on this stream waits for the task; the wait is encoded
  //    lazily, in front of that work (see deferred_waits_).
  deferred_waits_.push_back({fence_, done_value, "host-callback"});
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
  deferred_waits_.push_back({event->event_, v, "event"});
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
  if (v == 0) return absl::OkStatus();
  std::lock_guard<std::mutex> lock(mu_);
  deferred_waits_.push_back({other->fence_, v, "stream"});
  return absl::OkStatus();
}


void Device::RegisterStream(Stream* s) {
  std::lock_guard<std::mutex> lock(mu_);
  live_streams_.push_back(s);
}

void Device::UnregisterStream(Stream* s) {
  std::lock_guard<std::mutex> lock(mu_);
  live_streams_.erase(std::remove(live_streams_.begin(), live_streams_.end(), s),
                      live_streams_.end());
}

void Device::DumpStreams() {
  {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    uint64_t rss = 0;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
      rss = info.resident_size;
    }
    LOG(ERROR) << "  memory: pool " << (allocated_bytes() >> 20) << " MB of budget "
               << (memory_budget_ >> 20) << " MB, process RSS " << (rss >> 20)
               << " MB, reclaimable system memory "
               << (ReclaimableMemoryBytes() >> 20) << " MB";
  }
  // Hold mu_ so no stream can unregister (and be destroyed) meanwhile.
  std::lock_guard<std::mutex> lock(mu_);
  LOG(ERROR) << "  live streams on device " << ordinal_ << ": "
             << live_streams_.size();
  for (Stream* st : live_streams_) LOG(ERROR) << "    " << st->DebugState();
}

std::string Stream::DebugState() {
  // Called from a completion handler; do not take mu_ (the owner may hold it
  // while blocked in Synchronize). Scalars are read racily, for diagnostics;
  // containers only under their own locks.
  size_t host_tasks;
  {
    std::lock_guard<std::mutex> lock(work_mu_);
    host_tasks = work_.size();
  }
  std::string out = absl::StrFormat(
      "stream %p fence %p: next value %d, last committed %d, signaled %d, "
      "open cb %s, host tasks pending %d; last committed cb waited on:",
      this, fence_, fence_value_, last_committed_fence_value_,
      fence_->signaledValue(), cmd_ ? "yes" : "no", host_tasks);
  std::shared_ptr<const std::vector<std::shared_ptr<const KernelIdentity>>>
      kernels;
  {
    // The stream's references keep these events alive while diag_mu_ is held.
    std::lock_guard<std::mutex> lock(diag_mu_);
    for (const PendingWait& w : last_committed_waits_) {
      absl::StrAppendFormat(&out, " [%s %p value %d (signaled %d)]", w.kind,
                            w.event, w.value, w.event->signaledValue());
    }
    kernels = last_committed_kernels_;
  }
  absl::StrAppendFormat(&out, "; kernels in last committed cb (%d):",
                        kernels ? kernels->size() : 0);
  if (kernels) {
    for (const auto& k : *kernels) absl::StrAppend(&out, " ", k->name);
  }
  return out;
}

}  // namespace rt
}  // namespace metal_pjrt
