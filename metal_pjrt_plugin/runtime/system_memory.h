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

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_RUNTIME_SYSTEM_MEMORY_H_
