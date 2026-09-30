// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

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
//  - no fft op (XLA's FftThunk is cuFFT-only; JAX's fft lowers to metal$fft
//    on mtl). Checked here, after the simplifier has folded the platform
//    conditional of a multi-platform module (jax.export for ("cpu", "mtl"),
//    lax.platform_dependent), whose dead cpu branch may hold an fft.
//  - no rng op (jax.lax.rng_uniform): XLA generates its state update with a
//    legacy LLVM-IR kernel, which has no MSL translation.
// These run here rather than first in RunHloPasses because the simplifier
// removes some such ops (e.g. f32 -> f64 -> f32 chains) from programs that
// run. f64 here includes complex128. complex64 is supported (the emitter
// carries it as float2; MetalComplexDotExpander has turned its dots into
// real ones, so none is left here), except where it needs atomics: a
// scatter on complex without unique indices (XLA compare-and-swaps it in 64
// bits).
//  - every cuBLASLt GEMM is a plain "__cublas$lt$matmul" whose element types,
//    epilogue and (for f16/bf16, steel only) shape MetalBlasLt supports
//    (blas_lt_support.h).
// Returns an error naming the first offending op, so a violation fails at
// compile time instead of at run time or with wrong values.
absl::Status CheckPostGemmRewriter(const HloModule& module);

// What has to be refused before any pass runs: host offloading
// (jax.experimental.compute_on("device_host"), frontend attribute
// _xla_compute_type = "host"). XLA compiles such a region with XLA:CPU,
// which has none of the plugin's FFI handlers, after the plugin's passes
// have rewritten the ops inside it into metal$ calls, and it CHECK-fails
// when that compile fails (gpu_compiler.cc, "Failed to compile host
// executable").
absl::Status CheckBeforeOptimization(const HloModule& module);


}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_HLO_CHECKS_H_
