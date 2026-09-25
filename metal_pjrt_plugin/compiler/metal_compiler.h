#ifndef METAL_PJRT_PLUGIN_COMPILER_METAL_COMPILER_H_
#define METAL_PJRT_PLUGIN_COMPILER_METAL_COMPILER_H_

#include <memory>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// GpuCompiler for Apple Metal. Modeled on IntelGpuCompiler: the HLO pipeline
// is the stock one. Kernels are lowered to MSL by MetalKernelCompiler (see
// metal_pjrt_plugin/codegen), which CreateKernelCompiler hands to the
// emitters; the target-binary step hands back that MSL source or, for the
// constants module, a serialized constants container.
// Defaults the Metal backend needs regardless of what the client asked for:
//  - xla_gpu_enable_cub_radix_sort=false: SortRewriter would turn large sorts
//    into CUB FFI custom calls, which have no Metal handler.
//  - xla_gpu_enable_command_buffer cleared: no command buffer support (the
//    OneAPI capability disables them too; this makes it explicit).
//  - xla_gpu_enable_triton_gemm=false: Triton does not target Metal.
void ApplyMetalDefaults(DebugOptions& debug_options);

class MetalCompiler : public GpuCompiler {
 public:
  MetalCompiler();

  // Forces the debug options the Metal backend depends on, then runs the
  // stock GPU HLO pipeline.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  // Besides convolution canonicalization (a no-op), this is the last pre-layout
  // hook: it expands sort and triangular-solve, which have no Metal emitter.

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version,
      CompilationStats* compilation_stats) override;

  absl::Status AddConfigAssignerPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      stream_executor::StreamExecutor* stream_executor,
      const GpuTargetConfig* target_config, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store) override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      std::optional<int> shard_number) override;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;

  std::unique_ptr<KernelCompiler> CreateKernelCompiler(
      LlvmIrCompiler llvm_compiler,
      const se::DeviceDescription& device_description,
      const DebugOptions& debug_options,
      tsl::thread::ThreadPool* thread_pool) override;

  void AddPaddingForGpublasGemms(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version) override;

 private:
  MetalCompiler(const MetalCompiler&) = delete;
  MetalCompiler& operator=(const MetalCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_PLUGIN_COMPILER_METAL_COMPILER_H_
