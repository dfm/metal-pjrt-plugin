#ifndef METAL_PJRT_COMPILER_PASSES_HLO_CHECKS_H_
#define METAL_PJRT_COMPILER_PASSES_HLO_CHECKS_H_

#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {
namespace gpu {

// "<hlo name> (<JAX op_name>) at <file>:<line>", from the op's metadata and
// the module's stack frames, for error messages that point at user code.
std::string DescribeOp(const HloInstruction& instr);

// What the Metal backend relies on once GemmRewriter and
// MetalDotOperandUpcaster have run (end of OptimizeHloPostLayoutAssignment):
//  - no kDot has an operand narrower than its result (the loop emitter would
//    multiply in the narrow type);
//  - no arithmetic on f64 (Apple GPUs have no double precision); data
//    movement, calls and custom calls on f64 stay legal;
//  - no scatter on 64-bit elements with a read-modify-write combiner and
//    possibly repeated indices (needs 64-bit atomics).
// These run here rather than first in RunHloPasses because the simplifier
// removes some such ops (e.g. f32 -> f64 -> f32 chains) from programs that
// run. Complex arithmetic is not checked at all: the emitter lowers complex
// values inside a fusion (abs(fft(x)) runs); only complex kernel buffers fail,
// and that is only visible after fusion.
//  - every cuBLASLt GEMM is a plain "__cublas$lt$matmul" whose element types,
//    epilogue and (for f16/bf16, steel only) shape MetalBlasLt supports
//    (blas_lt_support.h).
// Returns an error naming the first offending op, so a violation fails at
// compile time instead of at run time or with wrong values.
absl::Status CheckPostGemmRewriter(const HloModule& module);


}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_HLO_CHECKS_H_
