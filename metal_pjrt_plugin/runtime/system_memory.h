// System memory queries. On unified-memory machines "GPU memory" is system
// RAM, so allocation policy must look at what the whole system has free.
#ifndef METAL_PJRT_PLUGIN_RUNTIME_SYSTEM_MEMORY_H_
#define METAL_PJRT_PLUGIN_RUNTIME_SYSTEM_MEMORY_H_

#include <cstdint>

namespace metal_pjrt {
namespace rt {

// Physical RAM in bytes.
uint64_t PhysicalMemoryBytes();

// Bytes the kernel can hand out without paging: free + inactive +
// speculative pages. Cheap (one mach call); safe to call per allocation.
uint64_t ReclaimableMemoryBytes();

// Allocation guard: on unified memory a GPU touching paged-out memory stalls
// until the watchdog fires, so allocations of 1 MB or more are refused when
// they would leave less than kSystemMemoryReserve reclaimable. Returns
// whether `size` fits; `*reclaimable` (if given) receives the current value.
inline constexpr uint64_t kSystemMemoryReserve = 512ull << 20;
bool FitsInSystemMemory(uint64_t size, uint64_t* reclaimable = nullptr);

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_RUNTIME_SYSTEM_MEMORY_H_
