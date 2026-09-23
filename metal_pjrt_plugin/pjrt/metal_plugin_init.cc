// Process-wide initialization for the Metal plugin: installs the MSL kernel
// emitter hook into XLA's MLIR kernel emitter. Everything else (platform,
// compiler, transfer manager, collectives, PJRT compiler) registers itself
// from its own translation unit.
#include "metal_pjrt_plugin/codegen/msl_kernel_hook.h"
#include "xla/stream_executor/platform/initialize.h"

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    metal_plugin_init, metal_pjrt::codegen::RegisterMetalKernelHook());
