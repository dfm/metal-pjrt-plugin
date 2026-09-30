// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/codegen/metal_kernel_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "metal_pjrt/codegen/msl_emitter.h"
#include "metal_pjrt/codegen/msl_llvm_bridge.h"
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
    int unroll_factor, xla::MlirKernelSource source) {
  // The module carries its own context (as upstream's CompileMlirToLlvm).
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(source).TakeModule();

  // Before any pass folds the thread ids away.
  const int threads_per_threadgroup =
      ThreadsPerThreadgroupFromRanges(module.get(), entry_function_name);

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
      EmitMslKernel(module.get(), entry_function_name, device,
                    threads_per_threadgroup);
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

MetalKernelCompiler::MetalKernelCompiler(
    xla::gpu::LlvmIrCompiler llvm_compiler,
    const stream_executor::DeviceDescription& device_info,
    const xla::DebugOptions& debug_options,
    tsl::thread::ThreadPool* thread_pool)
    : thread_pool_(thread_pool),
      inner_(
          [this, llvm_compiler = std::move(llvm_compiler)](
              llvm::Module& module,
              const stream_executor::DeviceDescription& descr,
              const xla::DebugOptions& opts) mutable
              -> absl::StatusOr<std::vector<uint8_t>> {
            if (pre_optimization_hook()) pre_optimization_hook()(module);
            return llvm_compiler(module, descr, opts);
          },
          device_info, debug_options, thread_pool) {}

xla::Future<std::unique_ptr<xla::gpu::Thunk>> MetalKernelCompiler::Compile(
    xla::gpu::Thunk::ThunkInfo thunk_info, xla::LlvmKernelSource kernel_source,
    const std::string& sanitized_kernel_name,
    const xla::emitters::KernelArguments& kernel_arguments,
    const xla::gpu::LaunchDimensions& launch_dimensions) {
  return inner_.Compile(std::move(thunk_info), std::move(kernel_source),
                        sanitized_kernel_name, kernel_arguments,
                        launch_dimensions);
}

xla::Future<xla::LlvmKernelSource> MetalKernelCompiler::CompileMlirToLlvm(
    const stream_executor::DeviceDescription& device,
    const xla::HloModule& hlo_module, const std::string& entry_function_name,
    int unroll_factor, xla::MlirKernelSource source,
    xla::gpu::BorrowedMlirContext borrowed_context) {
  if (!thread_pool_) {
    return CompileMlirToMsl(device, hlo_module, entry_function_name,
                            unroll_factor, std::move(source));
  }
  // `borrowed_context` owns the MLIRContext that `source` lives in; keep it
  // alive until the lowering is done (as CubinCustomKernelCompiler does).
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [source = std::move(source), device, &hlo_module, entry_function_name,
       unroll_factor,
       borrowed_context = std::move(borrowed_context)]() mutable {
        return CompileMlirToMsl(device, hlo_module, entry_function_name,
                                unroll_factor, std::move(source));
      });
}

xla::Future<xla::gpu::TritonWrapperResult>
MetalKernelCompiler::CompileTritonToLlvm(
    absl::string_view kernel_name, const xla::HloModule& hlo_module,
    const stream_executor::DeviceDescription& device_info,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    xla::gpu::TritonKernelSource triton_source,
    xla::gpu::BorrowedMlirContext borrowed_context, bool is_xla_fusion) {
  return inner_.CompileTritonToLlvm(
      kernel_name, hlo_module, device_info, block_level_parameters,
      target_triple, data_layout, std::move(triton_source),
      std::move(borrowed_context), is_xla_fusion);
}

xla::Future<std::vector<uint8_t>> MetalKernelCompiler::CompileToTargetBinary(
    xla::LlvmKernelSource kernel_source) {
  return inner_.CompileToTargetBinary(std::move(kernel_source));
}

}  // namespace metal_pjrt::codegen
