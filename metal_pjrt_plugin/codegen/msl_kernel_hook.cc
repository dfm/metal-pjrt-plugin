#include "metal_pjrt_plugin/codegen/msl_kernel_hook.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "metal_pjrt_plugin/codegen/msl_emitter.h"
#include "metal_pjrt_plugin/codegen/msl_llvm_bridge.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/transforms/pass_pipelines.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace metal_pjrt::codegen {

absl::StatusOr<xla::LlvmKernelSource> CompileMlirToMsl(
    const stream_executor::DeviceDescription& device,
    const xla::HloModule& hlo_module, const std::string& entry_function_name,
    int unroll_factor, mlir::MLIRContext& mlir_context,
    xla::MlirKernelSource source) {
  (void)mlir_context;  // The module carries its own context (as upstream).
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(source).TakeModule();

  // Same pipeline as xla::gpu::CompileMlirToLlvm, minus PDL, with the lowering
  // stopping before SCFToControlFlow.
  mlir::PassManager pm(module->getContext());
  bool should_verify =
      hlo_module.config().debug_options().xla_gpu_llvm_verification_level() >=
      1;
#ifndef NDEBUG
  should_verify = true;
#endif
  pm.enableVerifier(should_verify);
  xla::emitters::RegisterOptimizationPasses(pm);
  xla::gpu::AddLoopTransformationPasses(pm, device, unroll_factor);
  AddMslLoweringPasses(pm, device);
  {
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(module->getContext());
    (void)pm.run(module.get());
    absl::Status status = diagnostic_handler.consumeStatus();
    if (!status.ok()) {
      return absl::Status(status.code(),
                          absl::StrCat("MLIR lowering for Metal kernel '",
                                       entry_function_name,
                                       "' failed: ", status.message()));
    }
  }

  absl::StatusOr<MslKernel> kernel =
      EmitMslKernel(module.get(), entry_function_name, device);
  if (!kernel.ok()) return kernel.status();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>(entry_function_name, *llvm_context);
  if (absl::Status s = EmbedMslInLlvmModule(*llvm_module, *kernel); !s.ok()) {
    return s;
  }
  return xla::LlvmKernelSource(
      llvm::orc::ThreadSafeContext(std::move(llvm_context)),
      std::move(llvm_module));
}

void RegisterMetalKernelHook() { xla::gpu::SetMetalKernelHook(&CompileMlirToMsl); }

}  // namespace metal_pjrt::codegen
