// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/runtime/metal_runtime.h"
#include "metal_pjrt/runtime/buffer_cache.h"
#include "metal_pjrt/runtime/env.h"
#include "metal_pjrt/runtime/system_memory.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach-o/loader.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <limits>
#include <functional>
#include <thread>

#include "absl/cleanup/cleanup.h"
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
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
#include "metal_pjrt/kernels/runtime_builtins.metal.h"

namespace metal_pjrt {
namespace rt {

namespace {

// METAL_PJRT_TRACE=1 logs one line per committed command buffer with its op
// count and GPU execution time. Diagnostic only.
bool TraceEnabled() {
  static const bool enabled = EnvFlag("METAL_PJRT_TRACE");
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

// Waits on the host for `ev` to reach `v`, giving up once the device has
// failed (work committed after a reset may never run, and an unbounded wait
// leaves the process unable to exit, which is how the driver got wedged
// once).
// The error check is load-bearing, not an optimization to drop: after a
// failure, fences and events are force-signaled from the host, and an
// earlier buffer still in flight can then signal a LOWER value on the GPU
// (MTLSharedEvent signals are not monotonic), so a satisfied value may move
// backwards and a waiter must end on the error instead.
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

// "\n    #i <symbol>" per frame.
std::string FormatStack(const std::vector<void*>& frames) {
  std::string out;
  char name[512];
  for (size_t i = 0; i < frames.size(); ++i) {
    const char* sym =
        absl::Symbolize(frames[i], name, sizeof(name)) ? name : "?";
    absl::StrAppendFormat(&out, "\n    #%d %p %s", i, frames[i], sym);
  }
  return out;
}

// The error of a command buffer whose status is MTL::CommandBufferStatusError.
absl::Status CommandBufferError(MTL::CommandBuffer* cb, int ordinal) {
  const bool timeout =
      cb->error() != nullptr &&
      cb->error()->code() == MTL::CommandBufferErrorTimeout;
  return absl::InternalError(absl::StrFormat(
      "Metal: GPU command buffer failed on device %d: %s%s", ordinal,
      NSErrorToString(cb->error()),
      timeout ? ". The GPU watchdog stopped a computation that ran too "
                "long: split it into smaller jit calls or shrink the inputs"
              : ""));
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
  LOG(ERROR) << "  memory: live " << FormatBytes(m.live_bytes)
             << " + cached " << FormatBytes(m.cached_bytes) << " of budget "
             << FormatBytes(m.budget_bytes) << " (pressure level "
             << m.pressure << "), process RSS " << FormatBytes(rss)
             << ", reclaimable system memory "
             << FormatBytes(ReclaimableMemoryBytes());
}

// The budget refusal's remedy: the METAL_PJRT_MEMORY_FRACTION that raises the
// budget to the GPU's recommended working set, when that is above it.
std::string RaiseBudgetHint(uint64_t budget, uint64_t working_set) {
  const uint64_t base = std::min(PhysicalMemoryBytes() / 2, working_set);
  if (budget >= working_set || base == 0) return ".";
  const double f = std::floor(100.0 * working_set / base) / 100;
  if (f * base <= budget) return ".";
  return absl::StrFormat(
      ", or raise the budget: METAL_PJRT_MEMORY_FRACTION=%.2f gives %s (less "
      "memory for the rest of the system).",
      f, FormatBytes(std::min<uint64_t>(f * base, working_set)));
}

// Prefixes `context` to a non-OK status, keeping its code.
absl::Status Annotate(const absl::Status& s, absl::string_view context) {
  if (s.ok()) return s;
  return absl::Status(s.code(), absl::StrCat(context, ": ", s.message()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel

uint32_t Kernel::max_total_threads_per_threadgroup() const {
  const uint32_t pso =
      static_cast<uint32_t>(pso_->maxTotalThreadsPerThreadgroup());
  return declared_max_threads_ == 0 ? pso
                                    : std::min(pso, declared_max_threads_);
}

uint32_t Kernel::thread_execution_width() const {
  return static_cast<uint32_t>(pso_->threadExecutionWidth());
}

// ---------------------------------------------------------------------------
// Device

int Device::VisibleDeviceCount() {
  NS::Array* devices = MTL::CopyAllDevices();
  int n = devices ? static_cast<int>(devices->count()) : 0;
  if (devices) devices->release();
  return n;
}

namespace {
// The MTL::Device for `ordinal`, retained.
absl::StatusOr<MTL::Device*> CopyDevice(int ordinal) {
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
  return d;
}

DeviceInfo InfoOf(MTL::Device* d) {
  DeviceInfo info;
  info.name = d->name()->utf8String();
  info.max_buffer_length = d->maxBufferLength();
  info.recommended_working_set = d->recommendedMaxWorkingSetSize();
  info.max_threads_per_threadgroup =
      static_cast<uint32_t>(d->maxThreadsPerThreadgroup().width);
  info.threadgroup_memory_length =
      static_cast<uint32_t>(d->maxThreadgroupMemoryLength());
  info.gpu_family = HighestAppleFamily(d);
  info.simd_width = 32;
  return info;
}
}  // namespace

absl::StatusOr<DeviceInfo> Device::QueryInfo(int ordinal) {
  ABSL_ASSIGN_OR_RETURN(MTL::Device * d, CopyDevice(ordinal));
  DeviceInfo info = InfoOf(d);
  d->release();
  return info;
}

absl::StatusOr<std::unique_ptr<Device>> Device::Create(int ordinal) {
  ABSL_ASSIGN_OR_RETURN(MTL::Device * d, CopyDevice(ordinal));
  std::unique_ptr<Device> dev(new Device());
  dev->ordinal_ = ordinal;
  dev->device_ = d;
  dev->info_ = InfoOf(d);
  dev->quarantine_frees_ = EnvFlag("METAL_PJRT_DEBUG_FREE_QUARANTINE");
  const DeviceInfo& info = dev->info_;
  {
    // A ceiling, not a reservation: buffers are only mapped when used, and
    // the allocation-time guard refuses growth that would leave the system
    // without free memory, so a busy machine degrades to clean
    // RESOURCE_EXHAUSTED errors instead of swapping (a GPU stalling on
    // swapped-out pages is what trips the watchdog).
    uint64_t budget = PhysicalMemoryBytes() / 2;
    budget = std::min(budget, static_cast<uint64_t>(info.recommended_working_set));
    // METAL_PJRT_MEMORY_FRACTION scales it; above 1 it can grow up to the
    // working set.
    const char* v = std::getenv("METAL_PJRT_MEMORY_FRACTION");
    if (v != nullptr && v[0] != '\0') {
      double f = 0;
      if (!absl::SimpleAtod(v, &f) || !(f > 0) || !std::isfinite(f)) {
        LOG(WARNING) << "Ignoring METAL_PJRT_MEMORY_FRACTION=" << v
                     << " (not a number > 0); using 1, a budget of "
                     << FormatBytes(budget);
      } else {
        budget = std::min(static_cast<uint64_t>(budget * f),
                          static_cast<uint64_t>(info.recommended_working_set));
        if (f > 1) {
          LOG(WARNING) << "METAL_PJRT_MEMORY_FRACTION=" << v
                       << " is above 1: the memory budget grows past half "
                          "of RAM (capped at the GPU's recommended working "
                          "set), leaving less for the rest of the system";
        }
      }
    }
    dev->memory_budget_ = budget;
    LOG(INFO) << "Metal device " << ordinal << " (" << info.name
              << "): memory budget " << FormatBytes(budget) << ", "
              << FormatBytes(ReclaimableMemoryBytes()) << " reclaimable now";
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
    state_dir_ = absl::StrCat(base, "/metal-pjrt");
  }
  quarantine_strikes_ = static_cast<int>(std::min<uint64_t>(
      EnvUint("METAL_PJRT_QUARANTINE_STRIKES", quarantine_strikes_),
      std::numeric_limits<int>::max()));
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
  may_quarantine_ = !strikes_.empty();
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
    may_quarantine_ = !strikes_.empty();
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
             << " resets since boot are refused until a reboot or until that "
                "file is deleted (scripts/gpu_health.py --clear in a source "
                "checkout)";
}

bool Device::WaitForCompletionHandlers() {
  // After a failure Synchronize stops waiting for committed buffers, so a
  // completion handler (which uses this device) may still be pending. Wait
  // for it, bounded: after a reset it may never come.
  const auto deadline = std::chrono::steady_clock::now() + kHandlerWait;
  while (true) {
    {
      std::lock_guard<std::mutex> lock(in_flight_mu_);
      if (in_flight_.empty()) return true;
    }
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

Device::~Device() {
  if (!WaitForCompletionHandlers()) {
    LOG(ERROR) << "Metal device " << ordinal_
               << " destroyed with command buffers whose completion "
                  "handlers have not run";
  }
  {
    std::lock_guard<std::mutex> lock(g_devices_mu);
    g_devices.erase(std::find(g_devices.begin(), g_devices.end(), this));
  }
  if (memory_queue_ != nullptr) {
    auto timer = static_cast<dispatch_source_t>(trim_timer_);
    auto pressure = static_cast<dispatch_source_t>(pressure_source_);
    {
      // TrimCache (on the timer's queue) must not suspend the timer again.
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
      if (!trim_armed_) dispatch_resume(timer);  // suspended sources can't go
      trim_armed_ = true;
    }
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
  for (QuarantinedBuffer& q : quarantine_) q.a.buffer->release();
  if (!allocations_.empty()) {
    LOG(ERROR) << "Metal device " << ordinal_ << " destroyed with "
               << allocations_.size() << " live allocation(s) totalling "
               << allocated_bytes_ << " bytes; releasing them";
  }
  for (auto& [source, entry] : kernel_cache_) {
    for (auto& [key, kernel] : entry.kernels) kernel->pso()->release();
    entry.library->release();
  }
  for (auto& kv : allocations_) kv.second.buffer->release();
  for (auto& kv : host_allocations_) {
    if (kv.second.first != nullptr) {
      kv.second.first->release();
    } else {
      munmap(reinterpret_cast<void*>(kv.first), kv.second.second);
    }
  }
  if (device_) device_->release();
}

namespace {
std::mutex g_refusal_mu;
uint64_t g_refusal_count = 0;
uint64_t g_refusal_size = 0;
std::string g_refusal;
std::string g_refusal_executable;
thread_local uint64_t t_refusal = 0;

absl::Status RecordRefusal(uint64_t size, absl::Status s) {
  std::lock_guard<std::mutex> lock(g_refusal_mu);
  t_refusal = ++g_refusal_count;
  g_refusal_size = size;
  g_refusal = std::string(s.message());
  g_refusal_executable.clear();
  return s;
}
}  // namespace

std::string LastAllocationRefusal(uint64_t* size, std::string* executable) {
  std::lock_guard<std::mutex> lock(g_refusal_mu);
  *size = g_refusal_size;
  *executable = g_refusal_executable;
  return g_refusal;
}

uint64_t LastAllocationRefusalOnThisThread() { return t_refusal; }

void NameAllocationRefusal(uint64_t refusal, std::string executable) {
  std::lock_guard<std::mutex> lock(g_refusal_mu);
  if (refusal == g_refusal_count) g_refusal_executable = std::move(executable);
}

bool Device::CriticalAfterReleasingCache() {
  TrimCache(std::chrono::steady_clock::duration::zero());
  if (MemoryPressureLevel() < 2) return false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (cached_bytes_ == 0 || !error_.ok()) return true;
  }
  {
    std::unique_lock<std::mutex> lock(tickets_mu_);
    const uint64_t last = last_ticket_;
    tickets_cv_.wait_for(lock, kRefusalWait, [&] {
      return outstanding_.empty() || *outstanding_.begin() > last;
    });
  }
  TrimCache(std::chrono::steady_clock::duration::zero());
  return MemoryPressureLevel() >= 2;
}

absl::Status Device::CheckSystemMemory(uint64_t size) {
  if (size < (uint64_t{1} << 20)) return absl::OkStatus();
  const int level = MemoryPressureLevel();
  if (level == 0) return absl::OkStatus();
  static std::once_flag warned;
  std::call_once(warned, [] {
    LOG(WARNING) << "Metal: the system is short of memory (macOS memory "
                    "pressure is not normal); GPU work may slow down while "
                    "macOS compresses or swaps memory";
  });
  if (level < 2 || !CriticalAfterReleasingCache()) return absl::OkStatus();
  return absl::ResourceExhaustedError(absl::StrFormat(
      "Metal: allocating %s refused: the system is under critical memory "
      "pressure (system-wide, not this process's budget: it holds %s of "
      "its %s budget; device %d). Close other memory-heavy applications "
      "or use smaller arrays or batches",
      FormatBytes(size), FormatBytes(allocated_bytes()),
      FormatBytes(memory_budget_), ordinal_));
}

absl::StatusOr<Allocation> Device::Allocate(uint64_t size, Use use) {
  const uint64_t requested = size;
  if (size > info_.max_buffer_length) {
    return RecordRefusal(requested, absl::ResourceExhaustedError(absl::StrFormat(
        "Metal: allocating %s exceeds the largest buffer the device can "
        "create (maxBufferLength %s; device %d, %s)",
        FormatBytes(requested), FormatBytes(info_.max_buffer_length), ordinal_,
        info_.name)));
  }
  uint64_t length = BufferLength(size);
  if (length > info_.max_buffer_length) length = std::max<uint64_t>(size, 1);
  MTL::Buffer* buf = nullptr;
  std::vector<MTL::Buffer*> evicted;
  bool over_budget = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!quarantine_.empty()) DrainQuarantineLocked();
    auto it = FindCachedBuffer(
        cache_by_size_, length, std::min(2 * length, length + 2 * kPageBytes),
        use == Use::kHostWrite,
        use == Use::kHostWrite ? EndedBelow() : 0);
    if (it != cache_by_size_.end()) {
      auto node = it->second;
      buf = node->buffer;
      length = node->size;
      cached_bytes_ -= length;
      cache_by_size_.erase(it);
      cache_.erase(node);
      ++cache_hits_;
    } else {
      ++cache_misses_;
      const uint64_t ended = EndedBelow();
      while (!cache_.empty() && cache_.front().ticket < ended &&
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
    return RecordRefusal(requested, absl::ResourceExhaustedError(absl::StrCat(
        absl::StrFormat(
            "Metal: allocating %s refused: this process already holds %s of "
            "its %s memory budget (device %d; half of RAM, capped by the "
            "GPU's recommended working set). Use smaller arrays or batches",
            FormatBytes(requested), FormatBytes(allocated_bytes()),
            FormatBytes(memory_budget_), ordinal_),
        RaiseBudgetHint(memory_budget_, info_.recommended_working_set))));
  }
  if (buf == nullptr) {
    // Within the budget macOS pages as it would for any process; only at
    // critical pressure (where jetsam starts killing) is the allocation
    // refused. Our own cache goes first.
    if (absl::Status s = CheckSystemMemory(length); !s.ok()) {
      unreserve();
      return RecordRefusal(requested, std::move(s));
    }
    // Default (tracked) hazard mode: Metal orders dispatches touching the
    // same buffer for us.
    buf = device_->newBuffer(length, MTL::ResourceStorageModeShared);
    if (buf == nullptr) {
      unreserve();
      return RecordRefusal(requested, absl::ResourceExhaustedError(absl::StrFormat(
          "Metal: the driver could not allocate %s on device %d (%s): %s "
          "already allocated, recommended working set %s, maxBufferLength "
          "%s",
          FormatBytes(requested), ordinal_, info_.name,
          FormatBytes(allocated_bytes()),
          FormatBytes(info_.recommended_working_set),
          FormatBytes(info_.max_buffer_length))));
    }
  }
  void* ptr = buf->contents();
  {
    std::lock_guard<std::mutex> lock(mu_);
    const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    if (auto it = allocations_.find(key); it != allocations_.end()) {
      // Metal handed out an address we still hold: the table is wrong.
      LOG(ERROR) << "Metal device " << ordinal_ << ": a new " << length
                 << "-byte buffer starts at " << ptr
                 << ", already a live allocation (" << it->second.requested
                 << " bytes, generation " << it->second.generation << ")";
    }
    allocations_[key] = {buf, length, requested, ++generation_};
  }
  return Allocation{ptr, requested};
}

absl::Status Device::Deallocate(void* ptr, std::optional<uint64_t> size) {
  if (ptr == nullptr) return absl::OkStatus();
  bool under_pressure = false;
  {
    std::unique_lock<std::mutex> lock(mu_);
    const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    auto it = allocations_.find(key);
    if (it == allocations_.end()) {
      lock.unlock();
      return PointerBug(ptr, absl::StrFormat(
          "Deallocate(%p%s) is not the start of a live allocation", ptr,
          size ? absl::StrFormat(", %d bytes", *size) : ""));
    }
    if (size && *size != it->second.requested) {
      const LiveAllocation a = it->second;
      lock.unlock();
      return PointerBug(ptr, absl::StrFormat(
          "Deallocate(%p, %d bytes): the allocation there has %d bytes "
          "(generation %d)",
          ptr, *size, a.requested, a.generation));
    }
    const LiveAllocation a = it->second;
    allocated_bytes_ -= a.length;
    allocations_.erase(it);
    recent_frees_.push_back({key, a.requested, a.generation});
    if (recent_frees_.size() > kRecentFrees) recent_frees_.pop_front();
    uint64_t ticket;
    {
      std::lock_guard<std::mutex> tlock(tickets_mu_);
      ticket = last_ticket_;
    }
    if (quarantine_frees_) {
      QuarantinedBuffer q{a, key, std::chrono::steady_clock::now(),
                          ++free_seq_, ticket, std::vector<void*>(32)};
      q.free_stack.resize(absl::GetStackTrace(q.free_stack.data(), 32, 1));
      quarantine_.push_back(std::move(q));
      cached_bytes_ += a.length;  // counted as cached until it gets there
      DrainQuarantineLocked();
    } else {
      cache_.push_back({a.buffer, a.length, std::chrono::steady_clock::now(),
                        ticket});
      cache_by_size_.emplace(a.length, std::prev(cache_.end()));
      cached_bytes_ += a.length;
    }
    if (!trim_armed_ && trim_timer_ != nullptr) {
      dispatch_resume(static_cast<dispatch_source_t>(trim_timer_));
      trim_armed_ = true;
    }
    under_pressure = pressure_ > 0;
  }
  if (under_pressure) TrimCache(std::chrono::steady_clock::duration::zero());
  return absl::OkStatus();
}

void Device::DrainQuarantineLocked() {
  const auto now = std::chrono::steady_clock::now();
  while (!quarantine_.empty() &&
         free_seq_ - quarantine_.front().free_seq >= kQuarantineFrees &&
         now - quarantine_.front().freed >= kQuarantineTime) {
    QuarantinedBuffer& q = quarantine_.front();
    // Already counted in cached_bytes_.
    cache_.push_back({q.a.buffer, q.a.length, q.freed, q.ticket});
    cache_by_size_.emplace(q.a.length, std::prev(cache_.end()));
    quarantine_.pop_front();
  }
}

absl::Status Device::PointerBug(const void* ptr, const std::string& msg) {
  std::string known;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    for (auto r = recent_frees_.rbegin(); r != recent_frees_.rend(); ++r) {
      if (r->addr == addr) {
        known = absl::StrFormat(
            "; it was freed before (%d bytes, generation %d, %d frees ago): "
            "a double free",
            r->requested, r->generation,
            recent_frees_.rend() - r);
        break;
      }
    }
    for (const QuarantinedBuffer& q : quarantine_) {
      if (addr >= q.addr && addr < q.addr + q.a.length) {
        absl::StrAppendFormat(
            &known,
            "; it is inside a quarantined buffer freed %d ms ago (%d bytes "
            "at %#x, generation %d), freed at:%s",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - q.freed)
                .count(),
            q.a.requested, q.addr, q.a.generation, FormatStack(q.free_stack));
      }
    }
  }
  void* frames[32];
  const int depth = absl::GetStackTrace(frames, 32, 1);
  const std::string text = absl::StrFormat(
      "Metal device %d: %s%s (%s). This is a metal-pjrt-plugin bug; please "
      "report it with this log",
      ordinal_, msg, known, DescribeNearestAllocations(ptr));
  LOG(ERROR) << text << "\n  at:"
             << FormatStack(std::vector<void*>(frames, frames + depth));
  return absl::InternalError(text);
}

absl::StatusOr<void*> Device::AllocateHost(uint64_t size) {
  const uint64_t page = static_cast<uint64_t>(getpagesize());
  const uint64_t bytes = std::max<uint64_t>(size, 1);
  if (bytes > std::numeric_limits<uint64_t>::max() - page) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal: mapping %d bytes of host memory refused (device %d)", size,
        ordinal_));
  }
  const uint64_t mapped = (bytes + page - 1) / page * page;
  void* ptr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Metal: mapping %s of host memory failed (device %d)",
        FormatBytes(size), ordinal_));
  }
  // Beyond maxBufferLength the pages stay host-only (XLA's host staging
  // still works; a device copy of them is refused, as any pointer outside a
  // buffer is). XLA writes through the pool's pointer without a null check
  // (AllocateLinearizeDest), so only a failed mmap may fail here.
  MTL::Buffer* buf = nullptr;
  if (mapped <= info_.max_buffer_length) {
    buf = device_->newBuffer(
        ptr, mapped, MTL::ResourceStorageModeShared,
        ^(void* pointer, NS::UInteger length) { munmap(pointer, length); });
  }
  if (buf == nullptr) {
    LOG(WARNING) << "Metal device " << ordinal_ << ": " << FormatBytes(size)
                 << " of host memory is not reachable by the GPU";
  }
  std::lock_guard<std::mutex> lock(mu_);
  host_allocations_[reinterpret_cast<uintptr_t>(ptr)] = {buf, mapped};
  return ptr;
}

absl::Status Device::DeallocateHost(void* ptr) {
  MTL::Buffer* buf = nullptr;
  {
    std::unique_lock<std::mutex> lock(mu_);
    auto it = host_allocations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == host_allocations_.end()) {
      lock.unlock();
      return PointerBug(ptr, absl::StrFormat(
          "DeallocateHost(%p) is not the start of a host allocation", ptr));
    }
    buf = it->second.first;
    if (buf == nullptr) munmap(ptr, it->second.second);
    host_allocations_.erase(it);
  }
  // Command buffers that use it hold their own references; the pages go
  // with the last one.
  if (buf != nullptr) buf->release();
  return absl::OkStatus();
}

uint64_t Device::BeginWork() {
  std::lock_guard<std::mutex> lock(tickets_mu_);
  outstanding_.insert(++last_ticket_);
  return last_ticket_;
}

void Device::EndWork(uint64_t ticket) {
  {
    std::lock_guard<std::mutex> lock(tickets_mu_);
    outstanding_.erase(ticket);
  }
  tickets_cv_.notify_all();
}

uint64_t Device::EndedBelow() {
  std::lock_guard<std::mutex> lock(tickets_mu_);
  return outstanding_.empty() ? last_ticket_ + 1 : *outstanding_.begin();
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
  ++released_;
  out->push_back(it->buffer);
  cache_.erase(it);
}

void Device::ReleaseBuffers(const std::vector<MTL::Buffer*>& buffers) {
  for (MTL::Buffer* b : buffers) b->release();
}

void Device::TrimCache(std::chrono::steady_clock::duration min_idle) {
  std::vector<MTL::Buffer*> evicted;
  uint64_t bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    // Tickets grow with free order, so the front is the first to become
    // releasable.
    const uint64_t ended = EndedBelow();
    while (!cache_.empty() && now - cache_.front().freed >= min_idle &&
           cache_.front().ticket < ended) {
      bytes += cache_.front().size;
      EvictLocked(cache_.begin(), &evicted);
    }
    if (cache_.empty() && trim_armed_ && !stopping_) {
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
  m.released = released_;
  m.pressure = pressure_;
  return m;
}

std::string Device::DescribeNearestAllocations(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mu_);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto above = allocations_.upper_bound(addr);
  std::string out = absl::StrFormat("%d live allocations", allocations_.size());
  if (above != allocations_.begin()) {
    // `addr` may lie inside it (a range past its end is refused too).
    auto below = std::prev(above);
    absl::StrAppendFormat(&out, "; nearest at or below: %#x (%d bytes, %d bytes before)",
                          below->first, below->second.length,
                          addr - below->first);
  }
  if (above != allocations_.end()) {
    absl::StrAppendFormat(&out, "; nearest above: %#x (%d bytes, starts %d bytes after)",
                          above->first, above->second.length,
                          above->first - addr);
  }
  return out;
}

absl::StatusOr<BufferRef> Device::Resolve(const void* ptr) const {
  std::lock_guard<std::mutex> lock(mu_);
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  if (auto it = allocations_.upper_bound(addr); it != allocations_.begin()) {
    --it;
    if (addr < it->first + it->second.length) {
      return BufferRef{it->second.buffer, addr - it->first};
    }
  }
  if (auto it = host_allocations_.upper_bound(addr);
      it != host_allocations_.begin()) {
    --it;
    if (addr < it->first + it->second.second && it->second.first != nullptr) {
      return BufferRef{it->second.first, addr - it->first};
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
      why.code(),
      absl::StrCat("Metal device ", ordinal_,
                   " accepts no further GPU work in this process; restart "
                   "the Python process. Earlier GPU failure: ",
                   why.message()));
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
      SetError(CommandBufferError(f.cb, ordinal_));
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

absl::StatusOr<const Kernel*> Device::GetKernel(
    absl::string_view msl_source, absl::string_view function,
    absl::Span<const FunctionConstant> constants) {
  return GetKernelImpl(msl_source, function, constants, /*builtin=*/false,
                       /*pin=*/true);
}

absl::StatusOr<const Kernel*> Device::AcquireKernel(
    absl::string_view msl_source, absl::string_view function) {
  return GetKernelImpl(msl_source, function, {}, /*builtin=*/false,
                       /*pin=*/false);
}

void Device::HoldLocked(Kernel* k, bool pin) {
  if (pin) {
    k->pinned_ = true;
  } else {
    ++k->refs_;
  }
}

MTL::Library* Device::MaybeEraseLocked(absl::string_view source) {
  auto it = kernel_cache_.find(source);
  if (it == kernel_cache_.end() || !it->second.kernels.empty() ||
      it->second.pending > 0) {
    return nullptr;
  }
  MTL::Library* lib = it->second.library;
  kernel_cache_.erase(it);
  return lib;
}

void Device::ReleaseKernel(const Kernel* kernel) {
  MTL::ComputePipelineState* pso = nullptr;
  MTL::Library* lib = nullptr;
  {
    std::lock_guard<std::mutex> lock(kernel_mu_);
    Kernel* k = const_cast<Kernel*>(kernel);
    if (--k->refs_ > 0 || k->pinned_) return;
    pso = k->pso_;
    const absl::string_view source = k->source_;
    auto entry = kernel_cache_.find(source);
    entry->second.kernels.erase(entry->second.kernels.find(k->cache_key_));
    lib = MaybeEraseLocked(source);  // `source` dangles once erased
  }
  pso->release();
  if (lib != nullptr) lib->release();
}

Device::KernelCacheStats Device::kernel_cache_stats() {
  std::lock_guard<std::mutex> lock(kernel_mu_);
  KernelCacheStats stats;
  for (const auto& [source, entry] : kernel_cache_) {
    ++stats.libraries;
    stats.kernels += entry.kernels.size();
    stats.source_bytes += source.size();
  }
  return stats;
}

absl::Status Device::CheckQuarantine(const KernelIdentity& id) {
  if (id.builtin || quarantine_strikes_ == 0 ||
      !may_quarantine_.load(std::memory_order_acquire)) {
    return absl::OkStatus();
  }
  std::lock_guard<std::mutex> lock(mu_);
  auto strikes = strikes_.find(id.key);
  if (strikes == strikes_.end() || strikes->second < quarantine_strikes_) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Metal: kernel %s is quarantined: it was in the command buffer that "
      "timed out in %d GPU watchdog resets since boot. A reboot lifts the "
      "quarantine, as does deleting %s (scripts/gpu_health.py --clear in a "
      "source checkout); do that only once the cause is fixed (a changed "
      "kernel gets a new key anyway). METAL_PJRT_QUARANTINE_STRIKES=0 "
      "disables the quarantine",
      id.name, strikes->second, ResetLogPath(state_dir_)));
}

absl::StatusOr<const Kernel*> Device::GetKernelImpl(
    absl::string_view msl_source, absl::string_view function,
    absl::Span<const FunctionConstant> constants, bool builtin, bool pin) {
  // Kernels of one source by function name, plus the constants if any.
  std::string kernel_key(function);
  for (const FunctionConstant& c : constants) {
    absl::StrAppend(&kernel_key, "|", c.index,
                    c.type == FunctionConstant::Type::kBool ? "b" : "i",
                    c.value);
  }
  // `entry` (with `entry_source`, its key) is held by `pending` until this
  // call ends, so a concurrent ReleaseKernel cannot drop it.
  CachedLibrary* entry = nullptr;
  absl::string_view entry_source;
  Kernel* cached = nullptr;
  {
    std::lock_guard<std::mutex> lock(kernel_mu_);
    auto lib = kernel_cache_.find(msl_source);
    if (lib != kernel_cache_.end()) {
      auto k = lib->second.kernels.find(kernel_key);
      if (k != lib->second.kernels.end()) {
        cached = k->second.get();
        HoldLocked(cached, pin);
      } else {
        entry = &lib->second;
        entry_source = lib->first;
        ++entry->pending;
      }
    }
  }
  if (cached != nullptr) {
    absl::Status quarantined = CheckQuarantine(*cached->identity());
    if (!quarantined.ok()) {
      if (!pin) ReleaseKernel(cached);
      return quarantined;
    }
    return cached;
  }
  // The full source: its hash is the kernels' identity (a prelude change is
  // a new key), and it is what Metal compiles.
  const std::string source = codegen::ExpandMslPrelude(msl_source);
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  absl::Cleanup drain = [pool] { pool->release(); };
  if (entry == nullptr) {
    const std::string hash = HashString(source);
    NS::Error* err = nullptr;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(false);
    MTL::Library* lib = device_->newLibrary(Str(source), opts, &err);
    opts->release();
    if (lib == nullptr) {
      // The error lives in the autorelease pool; format it before draining.
      return absl::InternalError(absl::StrFormat(
          "Metal shader compilation failed (%d bytes of MSL, key %s): %s",
          source.size(), hash, NSErrorToString(err)));
    }
    std::lock_guard<std::mutex> lock(kernel_mu_);
    auto [it, inserted] = kernel_cache_.try_emplace(msl_source);
    if (inserted) {
      it->second.library = lib;
      it->second.hash = hash;
    } else {
      lib->release();  // lost a race; use the cached one
    }
    entry = &it->second;
    entry_source = it->first;
    ++entry->pending;
  }
  absl::Cleanup unpend = [this, entry, entry_source] {
    MTL::Library* lib = nullptr;
    {
      std::lock_guard<std::mutex> lock(kernel_mu_);
      --entry->pending;
      lib = MaybeEraseLocked(entry_source);
    }
    if (lib != nullptr) lib->release();
  };
  auto identity = std::make_shared<KernelIdentity>();
  identity->name = std::string(function);
  identity->key = absl::StrCat(entry->hash, ":", kernel_key);
  identity->builtin = builtin;
  ABSL_RETURN_IF_ERROR(CheckQuarantine(*identity));

  MTL::FunctionConstantValues* values = nullptr;
  if (!constants.empty()) {
    values = MTL::FunctionConstantValues::alloc()->init();
    for (const FunctionConstant& c : constants) {
      const bool b = c.value != 0;
      if (c.type == FunctionConstant::Type::kBool) {
        values->setConstantValue(&b, MTL::DataTypeBool, c.index);
      } else {
        values->setConstantValue(&c.value, MTL::DataTypeInt, c.index);
      }
    }
  }
  NS::Error* err = nullptr;
  MTL::Function* fn =
      values == nullptr
          ? entry->library->newFunction(Str(identity->name))
          : entry->library->newFunction(Str(identity->name), values, &err);
  if (values != nullptr) values->release();
  if (fn == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal function '", function, "' not found in library",
        err != nullptr ? absl::StrCat(": ", NSErrorToString(err)) : ""));
  }
  MTL::ComputePipelineState* pso = device_->newComputePipelineState(fn, &err);
  fn->release();
  if (pso == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Metal compute pipeline creation failed for '", function, "': ",
        NSErrorToString(err)));
  }
  const std::string name(function);
  auto kernel = std::make_unique<Kernel>(
      pso, std::move(identity), UsesArgumentBuffer(source, name),
      DeclaredMaxThreadsPerThreadgroup(source, name));
  kernel->source_ = entry_source;
  kernel->cache_key_ = kernel_key;
  std::lock_guard<std::mutex> lock(kernel_mu_);
  auto [it, inserted] = entry->kernels.try_emplace(kernel_key, nullptr);
  if (inserted) {
    it->second = std::move(kernel);
  } else {
    pso->release();  // lost a race; use the cached one
  }
  HoldLocked(it->second.get(), pin);
  return it->second.get();
}

namespace {
// Built-in kernels (kernels/runtime_builtins.metal).
constexpr const char* kBuiltinNames[] = {"xla_metal_fill32", "xla_metal_fill8",
                                         "xla_metal_copy16", "xla_metal_copy8"};
}  // namespace

absl::StatusOr<const Kernel*> Device::BuiltinKernel(Builtin kind) {
  // Launched by nearly every copy and fill: skip the cache's source lookup.
  std::lock_guard<std::mutex> lock(builtin_mu_);
  const Kernel*& slot = builtin_kernels_[static_cast<int>(kind)];
  if (slot == nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        slot, GetKernelImpl(kernels::kRuntimeBuiltinsMsl,
                            kBuiltinNames[static_cast<int>(kind)], {},
                            /*builtin=*/true, /*pin=*/true));
  }
  return slot;
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
  if (cmd_) {
    cmd_->release();
    device_->EndWork(cmd_ticket_);
  }
  for (auto& sv : pending_signals_) sv.first->release();
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
    // on_error runs as the task too: it must not call into this stream.
    current_host_task = {fence_, task.signal_value};
    if (!status.ok() && task.on_error) {
      task.on_error(status);
    } else {
      absl::Status own = task.fn();
      if (!own.ok() && task.on_error) {
        task.on_error(own);  // handled there
      } else if (!own.ok()) {
        device_->SetError(Annotate(own, "a Metal stream host task failed"));
      }
    }
    current_host_task = {};
    // A failed command buffer may have force-signaled the fence past this
    // value meanwhile; never move it backwards.
    if (fence_->signaledValue() < task.signal_value) {
      fence_->setSignaledValue(task.signal_value);
    }
    task.fn = nullptr;  // drops what it captured before the ticket ends
    device_->EndWork(task.ticket);
  }
}

absl::Status Stream::EnsureCommandBuffer() {
  // Checked for an open buffer too: after an error on another stream, no
  // more work goes into it (Commit drops it).
  ABSL_RETURN_IF_ERROR(device_->error());
  if (cmd_ != nullptr) {
    FlushDeferredWaits();
    return absl::OkStatus();
  }
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
  cmd_ticket_ = device_->BeginWork();
  ops_in_cmd_ = 0;
  threads_in_cmd_ = 0;
  flops_in_cmd_ = 0;
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
  if (!device_->error().ok()) {
    // The error is sticky and the GPU may have been reset: drop the buffer
    // instead of committing it, and end it as a failed buffer's handler
    // does, so waiters on its values wake up (and then see the error).
    // An earlier buffer still in flight may later signal the fence to a
    // lower value (see WaitForValueOnHost: host waits check the error).
    EndEncoder();
    const uint64_t v = ++fence_value_;
    last_committed_fence_value_ = v;
    if (fence_->signaledValue() < v) fence_->setSignaledValue(v);
    for (auto& sv : pending_signals_) {
      if (sv.first->signaledValue() < sv.second) {
        sv.first->setSignaledValue(sv.second);
      }
      sv.first->release();
    }
    pending_signals_.clear();
    pending_waits_.clear();
    pending_kernels_.clear();
    inject_error_ = absl::OkStatus();
    cmd_->release();
    cmd_ = nullptr;
    device_->EndWork(cmd_ticket_);
    ops_in_cmd_ = 0;
    threads_in_cmd_ = 0;
    flops_in_cmd_ = 0;
    return absl::OkStatus();
  }
  // Hold rule, plus: a buffer with no ops (RecordEvent after only waits,
  // e.g. a zero-byte device-to-device copy behind the compute sync point)
  // or on a transfer stream (set_waits_on_host) would sit on the GPU for as
  // long as it waits, which counts against the watchdog. Wait for such a
  // buffer's values on the host instead.
  for (const PendingWait& w : pending_waits_) {
    if (w.event->signaledValue() >= w.value) continue;
    if (!w.host_task && ops_in_cmd_ > 0 && !waits_on_host_) continue;
    ABSL_RETURN_IF_ERROR(CheckNotWaitingOnOwnHostTask(w.event, w.value));
    if (w.host_task) {
      device_->host_task_holds_.fetch_add(1, std::memory_order_relaxed);
    }
    ABSL_RETURN_IF_ERROR(WaitForValueOnHost(device_, w.event, w.value));
  }
  EndEncoder();
  absl::Status injected_error = std::move(inject_error_);
  inject_error_ = absl::OkStatus();
  // METAL_PJRT_FAIL_COMMAND_BUFFER=n fails the n-th command buffer this
  // process commits (testing error handling end to end).
  static const uint64_t fail_at = EnvUint("METAL_PJRT_FAIL_COMMAND_BUFFER", 0);
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
  std::vector<PendingWait> waits;
  waits.swap(pending_waits_);
  bool waits_on_gpu = false;
  for (const PendingWait& w : waits) {
    w.event->retain();
    if (w.event->signaledValue() >= w.value) continue;
    waits_on_gpu = true;
    if (w.host_task) {
      device_->unsignaled_host_task_waits_committed_.fetch_add(
          1, std::memory_order_relaxed);
    }
  }
  if (waits_on_gpu) {
    device_->gpu_waits_encoded_.fetch_add(1, std::memory_order_relaxed);
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
  const void* self = this;
  const uint64_t ticket = cmd_ticket_;
  device->AddInFlight(cmd_, fence_, v, injected_error);
  cmd_->addCompletedHandler([device, self, v, fence, signals, waits,
                             traced_ops, kernels, injected_error,
                             ticket](MTL::CommandBuffer* cb) {
    if (TraceEnabled()) {
      LOG(ERROR) << "[metal-trace] stream " << self << " cb#" << v
                 << " ops=" << traced_ops << " gpu_ms="
                 << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
    }
    absl::Status error = injected_error;
    const bool failed = cb->status() == MTL::CommandBufferStatusError;
    if (failed) error = CommandBufferError(cb, device->ordinal());
    if (!error.ok()) {
      LOG(ERROR) << error.message();
      // Diagnostics (--v=1): a timeout on a buffer with little GPU work
      // usually means it sat on a wait whose signal never came. Say which.
      if (VLOG_IS_ON(1)) {
        for (const PendingWait& w : waits) {
          VLOG(1) << "  this command buffer waited on " << w.kind
                  << " event " << w.event << " for value " << w.value
                  << "; its signaled value is now "
                  << w.event->signaledValue();
        }
        std::string names;
        for (const auto& k : *kernels) absl::StrAppend(&names, " ", k->name);
        VLOG(1) << "  ops in buffer: " << traced_ops << ", gpu time ms: "
                << (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3
                << ", kernels:" << names;
      }
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
    device->EndWork(ticket);
    // Last use of the device: ~Device waits for in_flight_ to empty.
    device->RemoveInFlight(cb);
    for (auto& sv : signals) sv.first->release();
    for (auto& w : waits) w.event->release();
    fence->release();
  });
  cmd_->commit();
  last_commit_time_ = std::chrono::steady_clock::now();
  // Keep the (retained) buffer until it completes; prune finished ones.
  in_flight_.push_back(cmd_);
  cmd_ = nullptr;
  ops_in_cmd_ = 0;
  threads_in_cmd_ = 0;
  flops_in_cmd_ = 0;
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  return Commit();
}

void Stream::set_waits_on_host(bool waits_on_host) {
  std::lock_guard<std::mutex> lock(mu_);
  waits_on_host_ = waits_on_host;
}

std::pair<uint64_t, uint64_t> Stream::FenceForTesting() {
  std::lock_guard<std::mutex> lock(mu_);
  return {last_committed_fence_value_, fence_->signaledValue()};
}

void Stream::FailNextCommandBufferForTesting(absl::Status error) {
  std::lock_guard<std::mutex> lock(mu_);
  inject_error_ = std::move(error);
}

absl::Status Stream::RefuseOwnHostTask() const {
  // Checked before taking mu_: a Commit on another thread may hold it while
  // it waits (hold rule) for this very task, and the task would block on it.
  if (current_host_task.fence != fence_) return absl::OkStatus();
  return absl::FailedPreconditionError(
      "a host task called into its own Metal stream; it could wait for "
      "itself");
}

absl::Status Stream::Synchronize() {
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
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

absl::Status Stream::FinishOp(uint64_t work, uint64_t flops) {
  threads_in_cmd_ += work;
  flops_in_cmd_ += std::min(flops, kMaxFlopsPerCommandBuffer);
  ++ops_in_cmd_;
  if (ops_in_cmd_ >= kMaxOpsPerCommandBuffer ||
      threads_in_cmd_ >= kMaxThreadsPerCommandBuffer ||
      flops_in_cmd_ >= kMaxFlopsPerCommandBuffer) {
    return Commit();
  }
  if (ops_in_cmd_ >= kEarlyCommitOps &&
      fence_->signaledValue() >= last_committed_fence_value_ &&
      std::chrono::steady_clock::now() - last_commit_time_ >=
          std::chrono::microseconds(kEarlyCommitIntervalUs)) {
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
                            uint32_t threadgroup_memory_bytes,
                            uint64_t flops) {
  ABSL_RETURN_IF_ERROR(ValidateLaunch(kernel, threads, args));
  if (kernel.uses_argument_buffer()) {
    return LaunchWithArgumentBuffer(kernel, threadgroups, threads, args,
                                    threadgroup_memory_bytes, flops);
  }
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  // Resolve before opening an encoder so a bad pointer encodes nothing.
  BufferRef refs[kMaxBufferArgs];
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
  return FinishOp(LaunchWork(threadgroups, threads), flops);
}

absl::Status Stream::LaunchWithArgumentBuffer(
    const Kernel& kernel, Dim3 threadgroups, Dim3 threads,
    absl::Span<const KernelArg> args, uint32_t threadgroup_memory_bytes,
    uint64_t flops) {
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  absl::InlinedVector<uint64_t, 64> addrs(std::max<size_t>(args.size(), 1), 0);
  absl::InlinedVector<MTL::Buffer*, 16> resident;
  for (size_t i = 0; i < args.size(); ++i) {
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
  return FinishOp(LaunchWork(threadgroups, threads), flops);
}

absl::Status Stream::EncodeExternal(
    std::function<absl::Status(void* mtl_command_buffer)> encode,
    uint64_t flops) {
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  EndEncoder();
  ABSL_RETURN_IF_ERROR(encode(static_cast<void*>(cmd_)));
  return FinishOp(0, flops);
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
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
  // The encoder is autoreleased; XLA threads have no pool of their own.
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
  if (blit == nullptr) {
    pool->release();
    return absl::InternalError(absl::StrFormat(
        "Metal blitCommandEncoder creation failed for a %d-byte copy", size));
  }
  blit->copyFromBuffer(src.buffer, src.offset, dst.buffer, dst.offset, size);
  blit->endEncoding();
  pool->release();
  return FinishOp(size / 4);  // about one thread per word
}

absl::Status Stream::Memset8(void* dst, uint8_t value, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
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
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::BlitCommandEncoder* blit = cmd_->blitCommandEncoder();
    if (blit == nullptr) {
      pool->release();
      return absl::InternalError(absl::StrFormat(
          "Metal blitCommandEncoder creation failed for a %d-byte fill", size));
    }
    blit->fillBuffer(dst.buffer, NS::Range(dst.offset, size), b);
    blit->endEncoding();
    pool->release();
    return FinishOp(size / 4);
  }
  // Word-granular when the range is 4-byte aligned; bytes otherwise.
  const bool words = size % 4 == 0 && dst.offset % 4 == 0;
  return EncodeBuiltin(words ? Device::Builtin::kFill32 : Device::Builtin::kFill8,
                       {dst}, &pattern, sizeof(pattern), words ? size / 4 : size);
}

absl::Status Stream::HostCallback(std::function<absl::Status()> fn,
                                  std::function<void(absl::Status)> on_error) {
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  // 1. Commit everything so far; it ends with a fence signal (value v). Waits
  //    still deferred on this stream become waits of the host task itself.
  //    After a failure the task is still enqueued: it runs or hands the
  //    error to on_error (see the header). That includes a failure seen
  //    while this commit held for an earlier host task: commit again (which
  //    drops the buffer) rather than lose the task. Only the self-wait
  //    refusal (no failure) is returned.
  absl::Status committed = Commit();
  if (!committed.ok()) {
    if (device_->error().ok()) return committed;
    ABSL_RETURN_IF_ERROR(Commit());
  }
  uint64_t done_value = ++fence_value_;
  uint64_t after_prior = last_committed_fence_value_;
  HostTask task{after_prior, std::move(fn), std::move(on_error), done_value,
                device_->BeginWork(), {}};
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

absl::Status Stream::CheckHostTransfer(const void* device_ptr, uint64_t size,
                                      absl::string_view what) {
  absl::StatusOr<BufferRef> ref = device_->Resolve(device_ptr);
  if (ref.ok() && ref->offset + size <= ref->buffer->length()) {
    return absl::OkStatus();
  }
  const std::string msg = absl::StrFormat(
      "Metal %s: device pointer %p (%d bytes) is %s a live allocation on "
      "device %d (%s); refusing the copy",
      what, device_ptr, size, ref.ok() ? "past the end of" : "not inside",
      device_->ordinal(), device_->DescribeNearestAllocations(device_ptr));
  LOG(ERROR) << msg;
  return absl::InternalError(absl::StrCat(
      msg,
      ". This is a metal-pjrt-plugin bug; please report it with the log "
      "above"));
}

absl::Status Stream::MemcpyHostToDevice(void* dst, const void* src,
                                        uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MemcpyHostToDevice(%p <- %p, %d bytes): null pointer", dst, src,
        size));
  }
  ABSL_RETURN_IF_ERROR(CheckHostTransfer(dst, size, "MemcpyHostToDevice"));
  {
    ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
    std::lock_guard<std::mutex> lock(mu_);
    if (IdleLocked()) {
      std::memcpy(dst, src, size);
      return absl::OkStatus();
    }
  }
  return HostCallback([dst, src, size]() {
    std::memcpy(dst, src, size);
    return absl::OkStatus();
  });
}

bool Stream::IdleLocked() {
  // CheckInFlight: a buffer that ran to its end may still have failed
  // (its handler not run yet); an inline copy must not read its output.
  if (cmd_ != nullptr || !device_->CheckInFlight(/*wait=*/false) ||
      !device_->error().ok()) {
    return false;
  }
  deferred_waits_.erase(
      std::remove_if(deferred_waits_.begin(), deferred_waits_.end(),
                     [](const PendingWait& w) {
                       return w.event->signaledValue() >= w.value;
                     }),
      deferred_waits_.end());
  return deferred_waits_.empty() && fence_->signaledValue() >= fence_value_;
}

absl::Status Stream::MemcpyDeviceToHost(void* dst, const void* src,
                                        uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MemcpyDeviceToHost(%p <- %p, %d bytes): null pointer", dst, src,
        size));
  }
  ABSL_RETURN_IF_ERROR(CheckHostTransfer(src, size, "MemcpyDeviceToHost"));
  {
    ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
    std::lock_guard<std::mutex> lock(mu_);
    if (IdleLocked()) {
      std::memcpy(dst, src, size);
      return absl::OkStatus();
    }
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  std::lock_guard<std::mutex> lock(mu_);
  ABSL_RETURN_IF_ERROR(EnsureCommandBuffer());
  uint64_t v;
  {
    std::lock_guard<std::mutex> elock(event->mu_);
    v = ++event->next_value_;
  }
  event->event_->retain();  // released by Commit's handler (or ~Stream)
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
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
  ABSL_RETURN_IF_ERROR(RefuseOwnHostTask());
  ABSL_RETURN_IF_ERROR(other->RefuseOwnHostTask());
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

// Test only: fakes the system memory-pressure level (0 normal, 1 warning,
// 2 critical) for allocations and runs every device's pressure handler as
// the dispatch source would. 0 turns the fake off; tests reset it.
__attribute__((visibility("default"))) void
metal_pjrt_testing_memory_pressure(int level) {
  using namespace metal_pjrt::rt;
  SetMemoryPressureForTesting(level);
  std::lock_guard<std::mutex> lock(g_devices_mu);
  for (Device* d : g_devices) d->OnMemoryPressure(level);
}

// Writes {live, cached, budget, cache hits, cache misses, pressure level,
// cached kernels, their MSL bytes} of device `ordinal` to out[0..7]; returns
// 0, or -1 if there is no such device.
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
    const Device::KernelCacheStats k = d->kernel_cache_stats();
    out[6] = k.kernels;
    out[7] = k.source_bytes;
    return 0;
  }
  return -1;
}

// Writes up to n of {gpu_waits_encoded, host_task_holds,
// unsignaled_host_task_waits_committed} of device `ordinal` to out; returns
// 0, or -1 if there is no such device. A separate hook so callers of
// metal_pjrt_memory_stats keep their 8-slot buffers.
__attribute__((visibility("default"))) int metal_pjrt_sync_stats(
    int ordinal, uint64_t* out, int n) {
  using namespace metal_pjrt::rt;
  std::lock_guard<std::mutex> lock(g_devices_mu);
  for (Device* d : g_devices) {
    if (d->ordinal() != ordinal) continue;
    const uint64_t v[] = {d->gpu_waits_encoded(), d->host_task_holds(),
                          d->unsignaled_host_task_waits_committed()};
    for (int i = 0; i < n && i < 3; ++i) out[i] = v[i];
    return 0;
  }
  return -1;
}

}  // extern "C"
