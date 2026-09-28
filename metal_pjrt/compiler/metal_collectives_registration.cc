#include <memory>

#include "metal_pjrt/pjrt/metal_platform_name.h"
#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/core/collectives/collectives_registry.h"

// Single-device backend: register the no-op collectives. The PJRT GPU client
// resolves them by the PJRT platform name (MetalName(), "openmetal") when it
// is created, collective thunks by the StreamExecutor platform name ("METAL",
// canonical "metal"), so register under both.
XLA_COLLECTIVES_REGISTER(xla::MetalName(), "stub", 1,
                         std::make_unique<xla::gpu::GpuCollectivesStub>());
XLA_COLLECTIVES_REGISTER("METAL", "stub", 1,
                         std::make_unique<xla::gpu::GpuCollectivesStub>());
