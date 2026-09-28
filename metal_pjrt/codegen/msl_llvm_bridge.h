// Carries MSL kernels through XLA's LLVM-module-based kernel pipeline.
//
// XLA's MlirKernelFusion::EmitLlvmModule expects an llvm::Module containing a
// function named after the kernel (it annotates it with launch bounds and a
// calling convention) and hands the module to GpuCompiler's
// CompileTargetBinary. For Metal the module is a stub: the kernel function is
// an empty `spir_kernel` with one addrspace(1) pointer per buffer argument, and
// the real code is MSL text stored in a constant global.
#ifndef METAL_PJRT_CODEGEN_MSL_LLVM_BRIDGE_H_
#define METAL_PJRT_CODEGEN_MSL_LLVM_BRIDGE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "llvm/IR/Module.h"
#include "metal_pjrt/codegen/msl_kernel.h"

namespace metal_pjrt::codegen {

// Globals holding MSL text are named kMslSourceGlobalPrefix + kernel name.
inline constexpr char kMslSourceGlobalPrefix[] = "__metal_msl_source.";
// Named metadata: one !{!"kernel", i32 num_buffer_args, ptr @source} per kernel.
inline constexpr char kMslNamedMetadata[] = "metal.msl";

// Adds the MSL source global, the kernel stub function and metadata to `m`.
absl::Status EmbedMslInLlvmModule(llvm::Module& m, const MslKernel& kernel);

// Returns the MSL source embedded in `m`, or nullopt if there is none. If
// several kernels were linked into one module their translation units are
// concatenated (ordered by kernel name). Kernel-specific symbols are prefixed
// with the kernel name, but each unit repeats the prelude without an include
// guard (removed as dead in 7275033), so only a one-kernel module yields a
// valid MSL translation unit; the compiler hands this one kernel at a time.
std::optional<std::string> ExtractMslFromLlvmModule(const llvm::Module& m);

struct EmbeddedMslKernel {
  std::string kernel_name;
  int num_buffer_args = 0;
};
// Lists the kernels recorded in the `metal.msl` metadata of `m`.
std::vector<EmbeddedMslKernel> ListMslKernels(const llvm::Module& m);

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_CODEGEN_MSL_LLVM_BRIDGE_H_
