// xla::gpu::KernelCompiler for Metal. MetalCompiler returns one from
// GpuCompiler::CreateKernelCompiler (see
// third_party/xla/patches/0002-gpu-compiler-kernel-compiler-factory.patch).
//
// CompileMlirToLlvm runs XLA's loop transformations and the lowering prefix up
// to (not including) SCFToControlFlow, converts the result to MSL and returns
// a stub llvm::Module carrying the MSL source (msl_llvm_bridge). Everything
// else (turning that module into a "binary" through
// MetalCompiler::CompileTargetBinary, and building the kernel thunk) is
// delegated to XLA's stock CubinCustomKernelCompiler.
#ifndef METAL_PJRT_PLUGIN_CODEGEN_METAL_KERNEL_COMPILER_H_
#define METAL_PJRT_PLUGIN_CODEGEN_METAL_KERNEL_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace metal_pjrt::codegen {

// Lowers one MLIR emitter kernel to MSL, wrapped in a stub llvm::Module.
absl::StatusOr<xla::LlvmKernelSource> CompileMlirToMsl(
    const stream_executor::DeviceDescription& device,
    const xla::HloModule& hlo_module, const std::string& entry_function_name,
    int unroll_factor, xla::MlirKernelSource source);

class MetalKernelCompiler final : public xla::gpu::KernelCompiler {
 public:
  MetalKernelCompiler(xla::gpu::LlvmIrCompiler llvm_compiler,
                      const stream_executor::DeviceDescription& device_info,
                      const xla::DebugOptions& debug_options,
                      tsl::thread::ThreadPool* thread_pool);

  // `inner_` captures `this`.
  MetalKernelCompiler(MetalKernelCompiler&&) = delete;
  MetalKernelCompiler& operator=(MetalKernelCompiler&&) = delete;

  xla::Future<std::unique_ptr<xla::gpu::Thunk>> Compile(
      xla::gpu::Thunk::ThunkInfo thunk_info,
      xla::LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const xla::emitters::KernelArguments& kernel_arguments,
      const xla::gpu::LaunchDimensions& launch_dimensions) override;

  xla::Future<xla::LlvmKernelSource> CompileMlirToLlvm(
      const stream_executor::DeviceDescription& device,
      const xla::HloModule& hlo_module, const std::string& entry_function_name,
      int unroll_factor, xla::MlirKernelSource source,
      xla::gpu::BorrowedMlirContext borrowed_context) override;

  xla::Future<xla::gpu::TritonWrapperResult> CompileTritonToLlvm(
      absl::string_view kernel_name, const xla::HloModule& hlo_module,
      const stream_executor::DeviceDescription& device_info,
      const xla::xtile::BlockLevelParameters& block_level_parameters,
      const llvm::Triple& target_triple, const std::string& data_layout,
      xla::gpu::TritonKernelSource triton_source,
      xla::gpu::BorrowedMlirContext borrowed_context,
      bool is_xla_fusion) override;

  xla::Future<std::vector<uint8_t>> CompileToTargetBinary(
      xla::LlvmKernelSource kernel_source) override;

 private:
  tsl::thread::ThreadPool* thread_pool_;
  // Handles everything downstream of the stub module. Its LlvmIrCompiler runs
  // this object's pre-optimization hook (GpuCompiler sets the hook on us, not
  // on `inner_`) before delegating to the real one.
  xla::gpu::CubinCustomKernelCompiler inner_;
};

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_PLUGIN_CODEGEN_METAL_KERNEL_COMPILER_H_
