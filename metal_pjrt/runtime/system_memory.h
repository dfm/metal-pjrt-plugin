// System memory queries. On unified-memory machines "GPU memory" is system
// RAM, so the allocator also looks at the system's memory pressure.
#ifndef METAL_PJRT_RUNTIME_SYSTEM_MEMORY_H_
#define METAL_PJRT_RUNTIME_SYSTEM_MEMORY_H_

#include <cstdint>
#include <string>

namespace metal_pjrt {
namespace rt {

// Physical RAM in bytes.
uint64_t PhysicalMemoryBytes();

// Bytes the kernel can hand out without paging: free + inactive +
// speculative + purgeable pages. Cheap (one mach call). Diagnostics and the
// device_put snapshot size only: macOS normally runs with little of it.
uint64_t ReclaimableMemoryBytes();

// The system's memory pressure now: 0 normal, 1 warning, 2 critical (the
// level jetsam acts on), from kern.memorystatus_vm_pressure_level: one
// sysctl (~1 us), cheap enough per allocation. A level set by
// SetMemoryPressureForTesting counts if higher.
int MemoryPressureLevel();
// Test only (metal_pjrt_testing_memory_pressure); 0 turns it off.
void SetMemoryPressureForTesting(int level);

// "12.3 MB" (MiB), or "N bytes" below 0.1 MB: the unit of every memory
// message.
std::string FormatBytes(uint64_t bytes);

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_SYSTEM_MEMORY_H_
