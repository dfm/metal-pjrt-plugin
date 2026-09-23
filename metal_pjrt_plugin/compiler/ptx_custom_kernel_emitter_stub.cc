// XLA's thunk emitter references EmitPtxCustomKernelThunk, whose definition
// is selected by CUDA/ROCm/SYCL build configuration. With none configured the
// symbol is undefined, so provide it here. PTX custom kernels never apply to
// Metal.
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {
namespace gpu {

class IrEmitterContext;

absl::StatusOr<std::unique_ptr<Thunk>> EmitPtxCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context) {
  return absl::UnimplementedError(
      "PTX custom kernels are not supported on the Metal backend");
}

}  // namespace gpu
}  // namespace xla
