#include <memory>

#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/core/collectives/collectives_registry.h"

// Single-device backend: register the no-op collectives so that the PJRT GPU
// client's GpuCollectives::Resolve("metal") succeeds.
XLA_COLLECTIVES_REGISTER("metal", "stub", 1,
                         std::make_unique<xla::gpu::GpuCollectivesStub>());
