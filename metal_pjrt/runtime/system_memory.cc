// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/runtime/system_memory.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>

#include "absl/strings/str_format.h"

namespace metal_pjrt {
namespace rt {

uint64_t PhysicalMemoryBytes() {
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) return 0;
  return mem;
}

uint64_t ReclaimableMemoryBytes() {
  // mach_host_self() adds a send-right reference on every call; one kept
  // for the process instead of one leaked per call (at 65534 references
  // the name overflows and this query would start failing).
  static const mach_port_t host = mach_host_self();
  vm_statistics64_data_t stats;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(host, HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&stats),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  vm_size_t page = 0;
  host_page_size(host, &page);
  uint64_t pages = static_cast<uint64_t>(stats.free_count) +
                   stats.inactive_count + stats.speculative_count +
                   stats.purgeable_count;
  return pages * page;
}

namespace {
std::atomic<int> g_test_pressure{0};
}  // namespace

int MemoryPressureLevel() {
  int level = 0;
  size_t len = sizeof(level);
  int system = 0;
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len,
                   nullptr, 0) == 0) {
    system = level >= 4 ? 2 : level >= 2 ? 1 : 0;  // 1, 2, 4 in the kernel
  }
  return std::max(system, g_test_pressure.load(std::memory_order_relaxed));
}

void SetMemoryPressureForTesting(int level) {
  g_test_pressure.store(level, std::memory_order_relaxed);
}

std::string FormatBytes(uint64_t bytes) {
  if (bytes < (uint64_t{1} << 20) / 10) {
    return absl::StrFormat("%d bytes", bytes);
  }
  return absl::StrFormat("%.1f MB", static_cast<double>(bytes) / (1 << 20));
}

}  // namespace rt
}  // namespace metal_pjrt
