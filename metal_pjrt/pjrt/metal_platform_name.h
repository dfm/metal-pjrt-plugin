#ifndef METAL_PJRT_PJRT_METAL_PLATFORM_NAME_H_
#define METAL_PJRT_PJRT_METAL_PLATFORM_NAME_H_

// MetalName() / MetalId() are added to xla/pjrt/pjrt_compiler.h by
// third_party/xla/patches/0001-metal-pjrt-identity.patch so that XLA's own
// IsGpuId() and the GPU client can refer to them. This header only exists so
// our code has one place to include.
#include "xla/pjrt/pjrt_compiler.h"

#endif  // METAL_PJRT_PJRT_METAL_PLATFORM_NAME_H_
