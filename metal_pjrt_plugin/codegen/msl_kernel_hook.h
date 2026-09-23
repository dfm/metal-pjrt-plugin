// Implementation of xla::gpu::MetalKernelHook (see
// xla_patch/mlir_kernel_emitter_hook.patch): runs XLA's loop transformations
// and the pre-SCFToControlFlow lowering prefix, converts the result to MSL and
// returns a stub llvm::Module carrying the MSL source.
#ifndef METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_HOOK_H_
#define METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_HOOK_H_

#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"

namespace metal_pjrt::codegen {

absl::StatusOr<xla::LlvmKernelSource> CompileMlirToMsl(
    const stream_executor::DeviceDescription& device,
    const xla::HloModule& hlo_module, const std::string& entry_function_name,
    int unroll_factor, mlir::MLIRContext& mlir_context,
    xla::MlirKernelSource source);

// Installs CompileMlirToMsl via xla::gpu::SetMetalKernelHook. Idempotent.
void RegisterMetalKernelHook();

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_HOOK_H_
