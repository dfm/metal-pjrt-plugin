#!/bin/bash
# Regenerates patches/0002-gpu-compiler-kernel-compiler-factory.patch: turns the
# hard-coded `CubinCustomKernelCompiler` in GpuCompiler::CompileToBackendResult
# into a protected virtual factory, so a GpuCompiler subclass (MetalCompiler)
# can supply its own KernelCompiler. Behavior for in-tree backends is unchanged.
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0002-gpu-compiler-kernel-compiler-factory.patch
WORK=$(mktemp -d)
FILES=(xla/service/gpu/gpu_compiler.h xla/service/gpu/gpu_compiler.cc)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b
perl -0pi -e 's|(#include "mlir/IR/MLIRContext.h"\n#include "xla/autotune_results.pb.h"\n)|$1#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"\n#include "xla/backends/gpu/codegen/kernel_compiler.h"\n|' $B/xla/service/gpu/gpu_compiler.h
perl -0pi -e 's|(  static std::unique_ptr<HloPassPipeline> GetCustomKernelRewriterPipeline\()|  // Creates the KernelCompiler that CompileToBackendResult hands to the\n  // emitters. `llvm_compiler` compiles one LLVM module to a target binary via\n  // CompileTargetBinary. The default is a CubinCustomKernelCompiler.\n  virtual std::unique_ptr<KernelCompiler> CreateKernelCompiler(\n      LlvmIrCompiler llvm_compiler,\n      const se::DeviceDescription& device_description,\n      const DebugOptions& debug_options, tsl::thread::ThreadPool* thread_pool);\n\n$1|' $B/xla/service/gpu/gpu_compiler.h
perl -0pi -e 's|(absl::StatusOr<GpuCompiler::CompileResultWithMetadata>\nGpuCompiler::CompileToBackendResult\()|std::unique_ptr<KernelCompiler> GpuCompiler::CreateKernelCompiler(\n    LlvmIrCompiler llvm_compiler,\n    const se::DeviceDescription& device_description,\n    const DebugOptions& debug_options, tsl::thread::ThreadPool* thread_pool) {\n  return std::make_unique<CubinCustomKernelCompiler>(\n      std::move(llvm_compiler), device_description, debug_options,\n      thread_pool);\n}\n\n$1|' $B/xla/service/gpu/gpu_compiler.cc
perl -0pi -e 's|    CubinCustomKernelCompiler kernel_compiler\(\n        std::move\(llvm_compiler\),\n        gpu_topology.gpu_target_config\(\).device_description,\n        module->config\(\).debug_options\(\), thread_pool.get_mutable\(\)\);\n    kernel_compiler.SetPreOptimizationHook|    std::unique_ptr<KernelCompiler> kernel_compiler = CreateKernelCompiler(\n        std::move(llvm_compiler),\n        gpu_topology.gpu_target_config().device_description,\n        module->config().debug_options(), thread_pool.get_mutable());\n    kernel_compiler->SetPreOptimizationHook|; s|            &kernel_compiler, std::move\(cpu_target_machine_options\),|            kernel_compiler.get(), std::move(cpu_target_machine_options),|' $B/xla/service/gpu/gpu_compiler.cc
cd $WORK && (diff -ruN a b > "$OUT" || true)
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
