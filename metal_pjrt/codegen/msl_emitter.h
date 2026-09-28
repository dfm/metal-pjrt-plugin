// MLIR (post-XLA-lowering, pre-SCFToControlFlow) -> Metal Shading Language.
//
// The input module is what XLA's MLIR kernel emitter produces after
// AddLoopTransformationPasses followed by AddMslLoweringPasses (the prefix of
// XLA's AddLoweringPasses that stops before SCFToControlFlow). At that point the
// IR contains func/scf/arith/math/vector/gpu ops plus the LLVM-dialect memory
// ops introduced by LowerTensors. EmitMslKernel converts that IR to EmitC
// (upstream Arith/SCF/Func->EmitC conversions plus custom patterns), prints it
// with MLIR's C++ emitter and wraps it in a `kernel` entry point.
#ifndef METAL_PJRT_CODEGEN_MSL_EMITTER_H_
#define METAL_PJRT_CODEGEN_MSL_EMITTER_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "metal_pjrt/codegen/msl_kernel.h"
#include "xla/stream_executor/device_description.h"

namespace metal_pjrt::codegen {

// Adds the prefix of xla::gpu::AddLoweringPasses (the Intel/OneAPI flavor) that
// precedes SCFToControlFlow. Run after AddLoopTransformationPasses.
void AddMslLoweringPasses(mlir::OpPassManager& pm,
                          const stream_executor::DeviceDescription& device);

// Threads per threadgroup of the launch `module` was emitted for: the product
// over x, y, z of (upper bound + 1) of the `xla.range` on the entry's
// gpu.thread_id ops, as XLA's emitters create them (one per dimension).
// Call before any pass runs (canonicalization folds unused or constant ids
// away). 0 when a dimension has no ranged thread_id op.
int ThreadsPerThreadgroupFromRanges(mlir::ModuleOp module,
                                    absl::string_view entry_function);

// Converts `module` (consumed: it is rewritten in place) to MSL. Returns
// UnimplementedError naming the first construct that cannot be translated.
// A positive `max_threads_per_threadgroup` is emitted as
// [[max_total_threads_per_threadgroup(N)]] (the runtime refuses larger
// threadgroups, see rt::DeclaredMaxThreadsPerThreadgroup).
absl::StatusOr<MslKernel> EmitMslKernel(
    mlir::ModuleOp module, absl::string_view entry_function,
    const stream_executor::DeviceDescription& device,
    int max_threads_per_threadgroup = 0);

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_CODEGEN_MSL_EMITTER_H_
