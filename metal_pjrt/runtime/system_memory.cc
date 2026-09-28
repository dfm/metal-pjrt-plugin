#include "metal_pjrt/runtime/system_memory.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <cstdint>

#include "metal_pjrt/runtime/env.h"

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

uint64_t SystemMemoryReserve() {
  static const uint64_t reserve = [] {
    return EnvUint("METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB", 512) << 20;
  }();
  return reserve;
}

bool FitsInSystemMemory(uint64_t size, uint64_t* reclaimable) {
  if (size < (1u << 20)) return true;
  const uint64_t r = ReclaimableMemoryBytes();
  if (reclaimable != nullptr) *reclaimable = r;
  const uint64_t reserve = SystemMemoryReserve();
  return r >= reserve && size <= r - reserve;
}

}  // namespace rt
}  // namespace metal_pjrt
