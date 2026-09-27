#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/runtime/system_memory.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach-o/loader.h>
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
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "absl/strings/ascii.h"

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

// Waits on the host for `ev` to reach `v`, giving up once the device has
// failed (work committed after a reset may never run, and an unbounded wait
// leaves the process unable to exit, which is how the driver got wedged
// once).
absl::Status WaitForValueOnHost(Device* device, MTL::SharedEvent* ev,
                                uint64_t v) {
  while (!ev->waitUntilSignaledValue(v, /*milliseconds=*/200)) {
    ABSL_RETURN_IF_ERROR(device->error());
  }
  return absl::OkStatus();
}

// The host task running on this thread, if any: its stream's fence and the
// value the task signals when done. Waiting on that fence for this value or
// later can never finish.
struct HostTaskContext {
  const MTL::SharedEvent* fence = nullptr;
  uint64_t value = 0;
};
thread_local HostTaskContext current_host_task;

absl::Status CheckNotWaitingOnOwnHostTask(const MTL::SharedEvent* ev,
                                          uint64_t v) {
  if (ev == nullptr || ev != current_host_task.fence ||
      v < current_host_task.value) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "a host task of a Metal stream waited for its own stream (value %d, the "
      "task signals %d when it returns); it would deadlock",
      v, current_host_task.value));
}

// The error of a command buffer whose status is MTL::CommandBufferStatusError.
absl::Status CommandBufferError(MTL::CommandBuffer* cb, int ordinal,
                                uint64_t fence_value) {
  return absl::InternalError(absl::StrFormat(
      "Metal command buffer failed on device %d (stream fence value %d): %s",
      ordinal, fence_value, NSErrorToString(cb->error())));
}

constexpr uint64_t kPageBytes = 16384;

// Buffer lengths are rounded so that similar requests share cached buffers:
// powers of two up to a page (at least 256 bytes), whole pages above.
uint64_t BufferLength(uint64_t size) {
  if (size > kPageBytes) return (size + kPageBytes - 1) / kPageBytes * kPageBytes;
  uint64_t n = 256;
  while (n < size) n *= 2;
  return n;
}

// Every live Device, for the C hooks at the end of this file.
std::mutex g_devices_mu;
std::vector<Device*> g_devices;

// Memory state, logged with a command buffer failure (a GPU stalling on
// swapped-out pages is what trips the watchdog).
void LogMemoryState(Device* device) {
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  uint64_t rss = 0;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    rss = info.resident_size;
  }
  const Device::MemoryStats m = device->memory_stats();
  LOG(ERROR) << "  memory: live " << (m.live_bytes >> 20) << " MB + cached "
             << (m.cached_bytes >> 20) << " MB of budget "
             << (m.budget_bytes >> 20) << " MB (pressure level " << m.pressure
             << ")"
             << " MB, process RSS " << (rss >> 20)
             << " MB, reclaimable system memory "
             << (ReclaimableMemoryBytes() >> 20) << " MB";
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
  const uint32_t pso =
      static_cast<uint32_t>(pso_->maxTotalThreadsPerThreadgroup());
  return declared_max_threads_ == 0 ? pso
                                    : std::min(pso, declared_max_threads_);
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
    // A ceiling, not a reservation: buffers are only mapped when used, and
    // the allocation-time guard refuses growth that would leave the system
    // without free memory, so a busy machine degrades to clean
    // RESOURCE_EXHAUSTED errors instead of swapping (a GPU stalling on
    // swapped-out pages is what trips the watchdog).
    uint64_t budget = PhysicalMemoryBytes() / 2;
    budget = std::min(budget, static_cast<uint64_t>(info.recommended_working_set));
    if (const char* v = std::getenv("JAX_OPENMETAL_MEMORY_FRACTION")) {
      const double f = std::atof(v);
      if (f > 0) {
        budget = std::min(static_cast<uint64_t>(budget * f),
                          static_cast<uint64_t>(info.recommended_working_set));
      }
    }
    dev->memory_budget_ = budget;
    LOG(INFO) << "Metal device " << ordinal << " (" << info.name
              << "): memory budget " << (budget >> 20) << " MB, "
              << (ReclaimableMemoryBytes() >> 20) << " MB reclaimable now";
  }  // All Apple GPUs; confirmed per-pipeline on creation.
  dev->LoadResetLog();
  dev->StartMemorySources();
  {
    std::lock_guard<std::mutex> lock(g_devices_mu);
    g_devices.push_back(dev.get());
  }
  return dev;
}

void Device::StartMemorySources() {
  dispatch_queue_t q =
      dispatch_queue_create("metal_pjrt.memory", DISPATCH_QUEUE_SERIAL);
  dispatch_source_t pressure = dispatch_source_create(
      DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
      DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
          DISPATCH_MEMORYPRESSURE_CRITICAL,
      q);
  dispatch_set_context(pressure, this);
  dispatch_source_set_event_handler_f(pressure, [](void* ctx) {
    Device* d = static_cast<Device*>(ctx);
    const uintptr_t m = dispatch_source_get_data(
        static_cast<dispatch_source_t>(d->pressure_source_));
    d->OnMemoryPressure((m & DISPATCH_MEMORYPRESSURE_CRITICAL) ? 2
                        : (m & DISPATCH_MEMORYPRESSURE_WARN)   ? 1
                                                               : 0);
  });
  dispatch_source_t timer =
      dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, q);
  const int64_t period =
      std::chrono::nanoseconds(kCacheIdleRelease).count() / 2;
  dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, period),
                            period, period / 2);
  dispatch_set_context(timer, this);
  dispatch_source_set_event_handler_f(timer, [](void* ctx) {
    static_cast<Device*>(ctx)->TrimCache(kCacheIdleRelease);
  });
  memory_queue_ = q;
  pressure_source_ = pressure;
  trim_timer_ = timer;
  dispatch_activate(pressure);
  dispatch_activate(timer);
  dispatch_suspend(timer);  // resumed while the cache is not empty
}

namespace {
std::string ResetLogPath(const std::string& state_dir) {
  return absl::StrCat(state_dir, "/gpu_resets.jsonl");
}

// Value of `"field":` in a one-line JSON record (number or string, no
// escapes), or empty.
absl::string_view JsonField(absl::string_view line, absl::string_view field) {
  const std::string tag = absl::StrCat("\"", field, "\":");
  size_t p = line.find(tag);
  if (p == absl::string_view::npos) return "";
  p += tag.size();
  if (p < line.size() && line[p] == '"') {
    size_t e = line.find('"', p + 1);
    return e == absl::string_view::npos ? "" : line.substr(p + 1, e - p - 1);
  }
  size_t e = line.find_first_of(",}", p);
  return e == absl::string_view::npos ? "" : line.substr(p, e - p);
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

std::string ImageUuid() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&ImageUuid), &info) == 0 ||
      info.dli_fbase == nullptr) {
    return "";
  }
  const auto* header = static_cast<const mach_header_64*>(info.dli_fbase);
  if (header->magic != MH_MAGIC_64) return "";
  const auto* cmd = reinterpret_cast<const load_command*>(header + 1);
  for (uint32_t i = 0; i < header->ncmds; ++i) {
    if (cmd->cmd == LC_UUID) {
      const auto* uuid = reinterpret_cast<const uuid_command*>(cmd);
      return std::string(reinterpret_cast<const char*>(uuid->uuid),
                         sizeof(uuid->uuid));
    }
    cmd = reinterpret_cast<const load_command*>(
        reinterpret_cast<const char*>(cmd) + cmd->cmdsize);
  }
  return "";
}

void Device::LoadResetLog() {
  build_ = absl::BytesToHexString(ImageUuid());
  const char* dir = std::getenv("METAL_PJRT_STATE_DIR");
  if (dir != nullptr && dir[0] != '\0') {
    state_dir_ = dir;
  } else {
    const char* home = std::getenv("HOME");
    std::string base = absl::StrCat(home != nullptr ? home : ".", "/.cache");
    state_dir_ = absl::StrCat(base, "/openmetal");
    // One-time migration from the directory used before the platform was
    // renamed "openmetal": copy its reset log (the quarantine is derived from
    // it) if there is none here yet. The old directory is left alone.
    std::string old_log = ResetLogPath(absl::StrCat(base, "/jax_metal"));
    std::error_code ec;
    if (!std::filesystem::exists(ResetLogPath(state_dir_), ec) &&
        std::filesystem::exists(old_log, ec)) {
      std::filesystem::create_directories(state_dir_, ec);
      std::filesystem::copy_file(old_log, ResetLogPath(state_dir_),
                                 std::filesystem::copy_options::skip_existing,
                                 ec);
    }
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
    // Each line is one reset: {"time":N,"build":"..",...,"kernels":[{"key":
    // "..",..},..]}. The keys are hex hashes plus a function name, so a
    // substring scan is enough; no JSON parser in the runtime.
    int64_t when = 0;
    absl::SimpleAtoi(JsonField(line, "time"), &when);
    if (when < boot) continue;
    // Strikes count per boot (the driver's state resets with it), for any
    // plugin build: builds change with every unrelated rebuild, and a kernel
    // whose code changed has a new key anyway. "build" is for diagnostics.
    ++resets_since_boot_;
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
      "{\"time\":%d,\"build\":\"%s\",\"pid\":%d,\"device\":%d,"
      "\"error\":\"%s\",\"builtins\":[",
      now, build_, static_cast<int>(getpid()), ordinal_, JsonEscape(cause));
  // Built-ins by name only (no "key", so they never count as strikes).
  std::vector<std::string> builtins;
  for (const auto& k : kernels) {
    if (k != nullptr && k->builtin &&
        std::find(builtins.begin(), builtins.end(), k->name) == builtins.end()) {
      absl::StrAppendFormat(&line, "%s\"%s\"", builtins.empty() ? "" : ",",
                            JsonEscape(k->name));
      builtins.push_back(k->name);
    }
  }
  line += "],\"kernels\":[";
  // Each kernel once, in launch order.
  std::vector<const KernelIdentity*> unique;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& k : kernels) {
      if (k == nullptr || k->builtin) continue;
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
             << " resets since boot are refused until a reboot or "
                "scripts/gpu_health.py --clear";
}

Device::~Device() {
  {
    std::lock_guard<std::mutex> lock(g_devices_mu);
    g_devices.erase(std::find(g_devices.begin(), g_devices.end(), this));
  }
  if (memory_queue_ != nullptr) {
    auto timer = static_cast<dispatch_source_t>(trim_timer_);
    auto pressure = static_cast<dispatch_source_t>(pressure_source_);
    if (!trim_armed_) dispatch_resume(timer);  // suspended sources can't go
    dispatch_source_cancel(timer);
    dispatch_source_cancel(pressure);
    // Wait out a handler that is already running.
    auto q = static_cast<dispatch_queue_t>(memory_queue_);
    dispatch_sync_f(q, nullptr, [](void*) {});
    dispatch_release(timer);
    dispatch_release(pressure);
    dispatch_release(q);
  }
  for (CachedBuffer& c : cache_) c.buffer->release();
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
  if (size > info_.max_buffer_length) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal allocation of %d bytes exceeds the device's maxBufferLength of "
        "%d bytes (device %d, %s)",
        requested, info_.max_buffer_length, ordinal_, info_.name));
  }
  uint64_t length = BufferLength(size);
  if (length > info_.max_buffer_length) length = std::max<uint64_t>(size, 1);
  MTL::Buffer* buf = nullptr;
  std::vector<MTL::Buffer*> evicted;
  bool over_budget = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_by_size_.lower_bound(length);
    if (it != cache_by_size_.end() &&
        it->first < std::min(2 * length, length + 2 * kPageBytes)) {
      auto node = it->second;
      buf = node->buffer;
      length = node->size;
      cached_bytes_ -= length;
      cache_by_size_.erase(it);
      cache_.erase(node);
      ++cache_hits_;
    } else {
      ++cache_misses_;
      while (!cache_.empty() &&
             allocated_bytes_ + cached_bytes_ + length > memory_budget_) {
        EvictLocked(cache_.begin(), &evicted);
      }
      over_budget = allocated_bytes_ + length > memory_budget_;
    }
    // Reserved now so that concurrent misses cannot both pass the budget.
    if (!over_budget) allocated_bytes_ += length;
  }
  ReleaseBuffers(evicted);
  auto unreserve = [&] {
    std::lock_guard<std::mutex> lock(mu_);
    allocated_bytes_ -= length;
  };
  if (over_budget) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal allocation of %d bytes refused: this process already holds %d "
        "bytes of its %d-byte memory budget (device %d; half of RAM, capped "
        "by the GPU's recommended working set, times "
        "JAX_OPENMETAL_MEMORY_FRACTION). Reduce the working set.",
        requested, allocated_bytes(), memory_budget_, ordinal_));
  }
  if (buf == nullptr) {
    // Refuse allocations that would push the system into swap (that is how
    // the machine got wedged once). Our own cache goes first.
    uint64_t reclaimable = 0;
    if (!FitsInSystemMemory(length, &reclaimable)) {
      TrimCache(std::chrono::steady_clock::duration::zero());
      if (!FitsInSystemMemory(length, &reclaimable)) {
        unreserve();
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Metal allocation of %d bytes refused: only %d bytes of system "
            "memory are reclaimable and %d bytes are reserved for the OS "
            "(device %d, %d bytes already allocated by this process, budget "
            "%d bytes). Reduce the working set or close other memory-heavy "
            "processes.",
            requested, reclaimable, kSystemMemoryReserve, ordinal_,
            allocated_bytes(), memory_budget_));
      }
    }
    // Default (tracked) hazard mode: Metal orders dispatches touching the
    // same buffer for us. Untracked mode with manual barriers is a later
    // optimization.
    buf = device_->newBuffer(length, MTL::ResourceStorageModeShared);
    if (buf == nullptr) {
      unreserve();
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Metal newBuffer failed for %d bytes on device %d (%s): %d bytes "
          "already allocated, recommended working set %d bytes, "
          "maxBufferLength %d bytes",
          requested, ordinal_, info_.name, allocated_bytes(),
          info_.recommended_working_set, info_.max_buffer_length));
    }
  }
  void* ptr = buf->contents();
  // METAL_PJRT_POISON_ALLOCATIONS=1 fills every allocation (fresh or
  // recycled) with 0xFF (NaN for floats, huge for ints) so reads of
  // uninitialized memory are visible and deterministic.
  static const bool poison = [] {
    const char* v = std::getenv("METAL_PJRT_POISON_ALLOCATIONS");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (poison) std::memset(ptr, 0xFF, length);
  {
    std::lock_guard<std::mutex> lock(mu_);
    allocations_[reinterpret_cast<uintptr_t>(ptr)] = {buf, length};
  }
  return Allocation{ptr, requested};
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
    const uint64_t length = it->second.second;
    allocated_bytes_ -= length;
    allocations_.erase(it);
    if (pressure_ == 0) {
      cache_.push_back({buf, length, std::chrono::steady_clock::now()});
      cache_by_size_.emplace(length, std::prev(cache_.end()));
      cached_bytes_ += length;
      if (!trim_armed_ && trim_timer_ != nullptr) {
        dispatch_resume(static_cast<dispatch_source_t>(trim_timer_));
        trim_armed_ = true;
      }
      return absl::OkStatus();
    }
  }
  ReleaseBuffers({buf});
  return absl::OkStatus();
}

void Device::EvictLocked(std::list<CachedBuffer>::iterator it,
                         std::vector<MTL::Buffer*>* out) {
  auto [lo, hi] = cache_by_size_.equal_range(it->size);
  for (auto m = lo; m != hi; ++m) {
    if (m->second == it) {
      cache_by_size_.erase(m);
      break;
    }
  }
  cached_bytes_ -= it->size;
  out->push_back(it->buffer);
  cache_.erase(it);
}

void Device::ReleaseBuffers(const std::vector<MTL::Buffer*>& buffers) {
  if (buffers.empty()) return;
  allocation_generation_.fetch_add(1, std::memory_order_acq_rel);
  for (MTL::Buffer* b : buffers) b->release();
}

void Device::TrimCache(std::chrono::steady_clock::duration min_idle) {
  std::vector<MTL::Buffer*> evicted;
  uint64_t bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    while (!cache_.empty() && now - cache_.front().freed >= min_idle) {
      bytes += cache_.front().size;
      EvictLocked(cache_.begin(), &evicted);
    }
    if (cache_.empty() && trim_armed_) {
      dispatch_suspend(static_cast<dispatch_source_t>(trim_timer_));
      trim_armed_ = false;
    }
  }
  ReleaseBuffers(evicted);
  if (!evicted.empty()) {
    VLOG(1) << "Metal device " << ordinal_ << ": released " << evicted.size()
            << " cached buffer(s), " << (bytes >> 20) << " MB";
  }
}

void Device::OnMemoryPressure(int level) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    pressure_ = level;
  }
  VLOG(1) << "Metal device " << ordinal_ << ": memory pressure level "
          << level;
  if (level > 0) TrimCache(std::chrono::steady_clock::duration::zero());
}

Device::MemoryStats Device::memory_stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  MemoryStats m;
  m.live_bytes = allocated_bytes_;
  m.cached_bytes = cached_bytes_;
  m.budget_bytes = memory_budget_;
  m.cache_hits = cache_hits_;
  m.cache_misses = cache_misses_;
  m.pressure = pressure_;
  return m;
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

absl::Status Device::error() const {
  std::lock_guard<std::mutex> lock(mu_);
  return error_;
}

void Device::SetError(const absl::Status& why) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!error_.ok()) return;
  error_ = absl::Status(
      why.code(), absl::StrCat(why.message(),
                               " (Metal device ", ordinal_,
                               "; no further GPU work is accepted in this "
                               "process, restart it)"));
}

void Device::AddInFlight(MTL::CommandBuffer* cb, MTL::SharedEvent* fence,
                         uint64_t value, absl::Status injected) {
  cb->retain();
  fence->retain();
  std::lock_guard<std::mutex> lock(in_flight_mu_);
  in_flight_.push_back({cb, fence, value, std::move(injected)});
}

void Device::RemoveInFlight(MTL::CommandBuffer* cb) {
  InFlight f{};
  {
    std::lock_guard<std::mutex> lock(in_flight_mu_);
    auto it = std::find_if(in_flight_.begin(), in_flight_.end(),
                           [cb](const InFlight& x) { return x.cb == cb; });
    if (it == in_flight_.end()) return;
    f = *it;
    in_flight_.erase(it);
  }
  f.cb->release();
  f.fence->release();
}

bool Device::CheckInFlight(bool wait) {
  // Collect under the lock, wait outside it (handlers take it to remove
  // their buffer).
  absl::InlinedVector<InFlight, 8> ended;
  {
    std::lock_guard<std::mutex> lock(in_flight_mu_);
    for (const InFlight& f : in_flight_) {
      const MTL::CommandBufferStatus st = f.cb->status();
      if (st == MTL::CommandBufferStatusCompleted && f.injected.ok()) continue;
      const bool settled = st == MTL::CommandBufferStatusCompleted ||
                         st == MTL::CommandBufferStatusError;
      if (settled || f.fence->signaledValue() >= f.value) {
        f.cb->retain();
        ended.push_back(f);
      }
    }
  }
  // The fence was read before this: a fence moved by a failure's
  // force-signal (not by the buffer itself) comes with the error set, and
  // then waiting is pointless and possibly unbounded.
  bool done = true;
  const bool failed = !error().ok();
  for (const InFlight& f : ended) {
    if (!failed && wait) f.cb->waitUntilCompleted();
    const MTL::CommandBufferStatus st = f.cb->status();
    if (st == MTL::CommandBufferStatusError) {
      SetError(CommandBufferError(f.cb, ordinal_, f.value));
    } else if (st != MTL::CommandBufferStatusCompleted) {
      done = false;
    } else if (!f.injected.ok()) {
      SetError(f.injected);
    }
    f.cb->release();
  }
  return done || failed;
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
    MTL::Library* library, const std::string& function, bool builtin) {
  if (library == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CreateKernel(", function, "): null library"));
  }
  std::string key =
      absl::StrCat(reinterpret_cast<uintptr_t>(library), ":", function);
  MTL::ComputePipelineState* pso = nullptr;
  auto identity = std::make_shared<KernelIdentity>();
  identity->name = function;
  identity->builtin = builtin;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto lk = library_keys_.find(library);
    identity->key = absl::StrCat(
        lk == library_keys_.end() ? "unknown" : lk->second, ":", function);
    auto strikes = strikes_.find(identity->key);
    if (!builtin && quarantine_strikes_ > 0 && strikes != strikes_.end() &&
        strikes->second >= quarantine_strikes_) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "kernel %s is quarantined: it was in the command buffer that timed "
          "out in %d GPU watchdog resets since boot (see "
          "%s/gpu_resets.jsonl). Fix the kernel (a changed kernel gets a new "
          "key); after a fix outside the kernel source, clear the log with "
          "scripts/gpu_health.py --clear. A reboot also lifts it; "
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
    ABSL_ASSIGN_OR_RETURN(
        slot, CreateKernel(lib, kBuiltinNames[static_cast<int>(kind)],
                           /*builtin=*/true));
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
  if (v == 0) return device_->error();
  // A failed command buffer force-signals its events (see Stream::Commit),
  // and the wait gives up once the device has failed, so this cannot hang.
  ABSL_RETURN_IF_ERROR(WaitForValueOnHost(device_, event_, v));
  device_->CheckInFlight(/*wait=*/true);
  return device_->error();
}

absl::StatusOr<bool> Event::Poll() {
  uint64_t v;
  {
    std::lock_guard<std::mutex> lock(mu_);
    v = value_;
  }
  if (v != 0 && event_->signaledValue() < v) return false;
  if (!device_->CheckInFlight(/*wait=*/false)) return false;
  ABSL_RETURN_IF_ERROR(device_->error());
  return true;
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
    // The waits give up once the device has failed.
    absl::Status status = WaitForValueOnHost(device_, fence_, task.wait_value);
    for (auto& [ev, value] : task.extra_waits) {
      if (status.ok()) status = WaitForValueOnHost(device_, ev, value);
      ev->release();
    }
    if (status.ok()) {
      device_->CheckInFlight(/*wait=*/true);
      status = device_->error();
    }
    // After a failure, a task with on_error does not run and on_error gets
    // the error. Without one, the task runs anyway (XLA's callbacks without
    // an error callback free memory or complete transfers).
    if (!status.ok() && task.on_error) {
      task.on_error(status);
    } else {
      current_host_task = {fence_, task.signal_value};
      absl::Status own = task.fn();
      current_host_task = {};
      if (!own.ok() && task.on_error) {
        task.on_error(own);  // handled there
      } else if (!own.ok()) {
        device_->SetError(Annotate(own, "a Metal stream host task failed"));
      }
    }
    // A failed command buffer may have force-signaled the fence past this
    // value meanwhile; never move it backwards.
    if (fence_->signaledValue() < task.signal_value) {
      fence_->setSignaledValue(task.signal_value);
    }
  }
}

absl::Status Stream::EnsureCommandBuffer() {
  if (cmd_ != nullptr) {
    FlushDeferredWaits();
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(device_->error());
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

absl::Status Stream::Commit() {
  if (cmd_ == nullptr) return absl::OkStatus();
  for (const PendingWait& w : pending_waits_) {
    if (!w.host_task || w.event->signaledValue() >= w.value) continue;
    ABSL_RETURN_IF_ERROR(CheckNotWaitingOnOwnHostTask(w.event, w.value));
    device_->host_task_holds_.fetch_add(1, std::memory_order_relaxed);
    ABSL_RETURN_IF_ERROR(WaitForValueOnHost(device_, w.event, w.value));
  }
  EndEncoder();
  absl::Status injected_error = std::move(inject_error_);
  inject_error_ = absl::OkStatus();
  // METAL_PJRT_FAIL_COMMAND_BUFFER=n fails the n-th command buffer this
  // process commits (testing error handling end to end).
  static const uint64_t fail_at = [] {
    const char* e = std::getenv("METAL_PJRT_FAIL_COMMAND_BUFFER");
    uint64_t n = 0;
    return e != nullptr && absl::SimpleAtoi(e, &n) ? n : 0;
  }();
  static std::atomic<uint64_t> commits{0};
  if (fail_at != 0 && commits.fetch_add(1) + 1 == fail_at) {
    injected_error = absl::InternalError(absl::StrFormat(
        "command buffer %d failed (METAL_PJRT_FAIL_COMMAND_BUFFER)", fail_at));
  }
  // Every command buffer ends by signaling the stream timeline, so
  // Synchronize/WaitForStream have a value to wait on, then its events.
  // Invariant: the fence signal comes BEFORE the event signals. A waiter
  // that sees an event value then knows the buffer's fence value is
  // signaled, i.e. the buffer ran to its end, which is what makes
  // Device::CheckInFlight wait for (and check) that buffer's status.
  // An injected failure encodes no signals: its handler force-signals them.
  const uint64_t v = ++fence_value_;
  std::vector<std::pair<MTL::SharedEvent*, uint64_t>> signals;
  signals.swap(pending_signals_);
  if (injected_error.ok()) {
    cmd_->encodeSignalEvent(fence_, v);
    for (auto& sv : signals) cmd_->encodeSignalEvent(sv.first, sv.second);
  }
  for (auto& sv : signals) sv.first->retain();
  std::vector<PendingWait> waits;
  waits.swap(pending_waits_);
  for (const PendingWait& w : waits) {
    w.event->retain();
    if (w.host_task && w.event->signaledValue() < w.value) {
      device_->unsignaled_host_task_waits_committed_.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
  auto kernels =
      std::make_shared<const std::vector<std::shared_ptr<const KernelIdentity>>>(
          std::move(pending_kernels_));
  pending_kernels_.clear();
  MTL::SharedEvent* fence = fence_;
  fence->retain();
  const int traced_ops = ops_in_cmd_;
  // The handler must not touch `this`: after a failure ~Stream stops
  // waiting for in-flight buffers, whose handlers may run later.
  Device* device = device_;
  std::shared_ptr<CompletionState> state = completion_;
  const void* self = this;
  state->pending.fetch_add(1, std::memory_order_relaxed);
  device->AddInFlight(cmd_, fence_, v, injected_error);
  cmd_->addCompletedHandler([device, state, self, v, fence, signals, waits,
                             traced_ops, kernels,
                             injected_error](MTL::CommandBuffer* cb) {
    if (TraceEnabled()) {
      LOG(ERROR) << "[metal-trace] stream " << self << " cb#" << v
                 << " ops=" << traced_ops << " gpu_ms="
                 << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
    }
    absl::Status error = injected_error;
    const bool failed = cb->status() == MTL::CommandBufferStatusError;
    if (failed) error = CommandBufferError(cb, device->ordinal(), v);
    if (!error.ok()) {
      LOG(ERROR) << error.message();
      // Diagnostics: a timeout on a buffer with little GPU work usually
      // means it sat on a wait whose signal never came. Say which.
      for (const PendingWait& w : waits) {
        LOG(ERROR) << "  this command buffer waited on " << w.kind
                   << " event " << w.event << " for value " << w.value
                   << "; its signaled value is now "
                   << w.event->signaledValue();
      }
      std::string names;
      for (const auto& k : *kernels) absl::StrAppend(&names, " ", k->name);
      LOG(ERROR) << "  ops in buffer: " << traced_ops << ", gpu time ms: "
                 << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3
                 << ", kernels:" << names;
      LogMemoryState(device);
      // Watchdog timeouts and revoked/removed devices mean the GPU was
      // reset underneath us: record the suspects (quarantine).
      const long code = failed && cb->error() ? cb->error()->code() : 0;
      if (code == MTL::CommandBufferErrorTimeout ||
          code == MTL::CommandBufferErrorAccessRevoked ||
          code == MTL::CommandBufferErrorDeviceRemoved) {
        device->RecordReset(error.message(), *kernels);
      }
      // Before the force-signals below, so waiters woken by them see it.
      device->SetError(error);
      if (fence->signaledValue() < v) fence->setSignaledValue(v);
      for (auto& sv : signals) {
        if (sv.first->signaledValue() < sv.second) {
          sv.first->setSignaledValue(sv.second);
        }
      }
    }
    device->RemoveInFlight(cb);
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

std::pair<uint64_t, uint64_t> Stream::FenceForTesting() {
  std::lock_guard<std::mutex> lock(mu_);
  return {last_committed_fence_value_, fence_->signaledValue()};
}

void Stream::FailNextCommandBufferForTesting(absl::Status error) {
  std::lock_guard<std::mutex> lock(mu_);
  inject_error_ = std::move(error);
}

absl::Status Stream::Synchronize() {
  if (current_host_task.fence == fence_) {
    return absl::FailedPreconditionError(
        "Synchronize called from a host task of the same Metal stream; it "
        "would wait for itself");
  }
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(Commit());
  // Wait for completion (not just the fence signal) so resources referenced
  // by these command buffers may be freed by the caller right away.
  for (MTL::CommandBuffer* cb : in_flight_) {
    // Bounded wait: after a failure (a reset), committed work may never run.
    while (true) {
      MTL::CommandBufferStatus st = cb->status();
      if (st == MTL::CommandBufferStatusCompleted ||
          st == MTL::CommandBufferStatusError) {
        break;
      }
      // Read the fence before the error: a fence moved by a failure's
      // force-signal comes with the error already set.
      const bool ended = fence_->signaledValue() >= last_committed_fence_value_;
      if (!device_->error().ok()) break;  // abandon it
      if (ended) {
        // The buffer's last action (the fence signal) has run, so its status
        // flips to Completed as soon as the completion handler fires; waiting
        // for that cannot hang. (A timed sleep here cost ~1 ms per
        // Synchronize and dominated sync-heavy programs.)
        cb->waitUntilCompleted();
        break;
      }
      fence_->waitUntilSignaledValue(last_committed_fence_value_, 200);
    }
    cb->release();
  }
  in_flight_.clear();
  // Waits not yet encoded are satisfied on the host instead of by a
  // wait-only command buffer (which would count against the GPU watchdog).
  absl::Status status;
  for (const PendingWait& w : deferred_waits_) {
    if (!status.ok()) break;
    status = CheckNotWaitingOnOwnHostTask(w.event, w.value);
    if (status.ok()) status = WaitForValueOnHost(device_, w.event, w.value);
  }
  deferred_waits_.clear();
  ABSL_RETURN_IF_ERROR(status);
  device_->CheckInFlight(/*wait=*/true);
  return device_->error();
}

bool UsesArgumentBuffer(const std::string& msl_source,
                        const std::string& kernel_name) {
  return msl_source.find(absl::StrCat(kArgumentBufferMarker, "\nkernel void ",
                                      kernel_name, "(")) != std::string::npos;
}

uint32_t DeclaredMaxThreadsPerThreadgroup(const std::string& msl_source,
                                          const std::string& kernel_name) {
  const size_t k =
      msl_source.find(absl::StrCat("kernel void ", kernel_name, "("));
  if (k == std::string::npos) return 0;
  constexpr absl::string_view kAttr = "[[max_total_threads_per_threadgroup(";
  const size_t a = msl_source.rfind(kAttr, k);
  if (a == std::string::npos) return 0;
  const size_t close = msl_source.find(")]]", a);
  if (close == std::string::npos || close > k) return 0;
  // Only whitespace and the argument-buffer marker may separate the two.
  std::string between = msl_source.substr(close + 3, k - close - 3);
  absl::StrReplaceAll({{kArgumentBufferMarker, ""}}, &between);
  if (!absl::StripAsciiWhitespace(between).empty()) return 0;
  uint32_t n = 0;
  if (!absl::SimpleAtoi(absl::string_view(msl_source).substr(
                            a + kAttr.size(), close - a - kAttr.size()),
                        &n)) {
    return 0;
  }
  return n;
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

absl::Status Stream::HostCallback(std::function<absl::Status()> fn,
                                  std::function<void(absl::Status)> on_error) {
  std::lock_guard<std::mutex> lock(mu_);
  // 1. Commit everything so far; it ends with a fence signal (value v). Waits
  //    still deferred on this stream become waits of the host task itself.
  //    After a failure the task is still enqueued: it runs or hands the
  //    error to on_error (see the header).
  ABSL_RETURN_IF_ERROR(Commit());
  uint64_t done_value = ++fence_value_;
  uint64_t after_prior = last_committed_fence_value_;
  HostTask task{after_prior, std::move(fn), std::move(on_error), done_value,
                {}};
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
  deferred_waits_.push_back({fence_, done_value, "host-callback", true});
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
  return HostCallback([dst, src, size]() {
    std::memcpy(dst, src, size);
    return absl::OkStatus();
  });
}

absl::Status Stream::MemcpyDeviceToHost(void* dst, const void* src,
                                        uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MemcpyDeviceToHost(%p <- %p, %d bytes): null pointer", dst, src,
        size));
  }
  return HostCallback([dst, src, size]() {
    std::memcpy(dst, src, size);
    return absl::OkStatus();
  });
}

absl::Status Stream::RecordEvent(Event* event) {
  if (event == nullptr) {
    return absl::InvalidArgumentError("RecordEvent: null event");
  }
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  uint64_t v;
  {
    std::lock_guard<std::mutex> elock(event->mu_);
    v = ++event->next_value_;
  }
  pending_signals_.emplace_back(event->event_, v);  // encoded by Commit
  // Commit so a host or another stream waiting on the event can make
  // progress, and publish the value only then: Commit may first wait for a
  // host task (hold rule), and a wait on the value before its signaling
  // buffer is committed would sit on the GPU. Waiters on a published value
  // therefore never wait for unfinished host work.
  ABSL_RETURN_IF_ERROR(Commit());
  std::lock_guard<std::mutex> elock(event->mu_);
  event->value_ = std::max(event->value_, v);
  return absl::OkStatus();
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
  // Wait for the highest value `other` has issued: after its Commit that is
  // either its last command buffer or a host task enqueued after it (whose
  // value is signaled by the host, hence the hold rule applies). Values are
  // monotonic in stream order: later work waits for earlier host tasks.
  // Also inherit the waits `other` has not encoded yet (it may have launched
  // nothing since it was told to wait): waiting for `other` means waiting
  // for everything it is ordered after. Same lifetime rule as any deferred
  // wait (the events' owners outlive the wait).
  uint64_t v;
  bool host_task;
  std::vector<PendingWait> inherited;
  {
    std::lock_guard<std::mutex> olock(other->mu_);
    ABSL_RETURN_IF_ERROR(other->Commit());
    v = other->fence_value_;
    host_task = v > other->last_committed_fence_value_;
    for (const PendingWait& w : other->deferred_waits_) {
      if (w.event != other->fence_) inherited.push_back(w);  // else <= v
    }
  }
  std::lock_guard<std::mutex> lock(mu_);
  deferred_waits_.insert(deferred_waits_.end(), inherited.begin(),
                         inherited.end());
  if (v == 0) return absl::OkStatus();
  deferred_waits_.push_back({other->fence_, v, "stream", host_task});
  return absl::OkStatus();
}

}  // namespace rt
}  // namespace metal_pjrt

// Test and diagnostic hooks, exported from the plugin dylib (see
// pjrt/BUILD.bazel) for ctypes.
extern "C" {

// Runs every device's memory-pressure handler as the dispatch source would
// (0 normal, 1 warning, 2 critical).
__attribute__((visibility("default"))) void metal_pjrt_memory_pressure(
    int level) {
  using namespace metal_pjrt::rt;
  std::lock_guard<std::mutex> lock(g_devices_mu);
  for (Device* d : g_devices) d->OnMemoryPressure(level);
}

// Writes {live, cached, budget, cache hits, cache misses, pressure level} of
// device `ordinal` to out[0..5]; returns 0, or -1 if there is no such device.
__attribute__((visibility("default"))) int metal_pjrt_memory_stats(
    int ordinal, uint64_t* out) {
  using namespace metal_pjrt::rt;
  std::lock_guard<std::mutex> lock(g_devices_mu);
  for (Device* d : g_devices) {
    if (d->ordinal() != ordinal) continue;
    const Device::MemoryStats m = d->memory_stats();
    out[0] = m.live_bytes;
    out[1] = m.cached_bytes;
    out[2] = m.budget_bytes;
    out[3] = m.cache_hits;
    out[4] = m.cache_misses;
    out[5] = static_cast<uint64_t>(m.pressure);
    return 0;
  }
  return -1;
}

}  // extern "C"
