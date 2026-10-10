// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_CONV_REWRITER_H_
#define METAL_PJRT_COMPILER_PASSES_CONV_REWRITER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Replaces 1-D and 2-D convolutions with a "metal$conv" FFI custom call
// (metal_pjrt/ffi/conv_ffi.cc) on MLX's steel kernels (metal_pjrt/conv/
// conv.h) instead of the loop emitter's one thread per output element.
//
// Runs in MetalCompiler::OptimizeHloConvolutionCanonicalization (before
// layout assignment). Each convolution becomes
//
//   transpose(operands) -> custom-call "metal$conv" -> (out, u8[workspace])
//                       -> get-tuple-element 0 -> transpose back
//
// with the operands in the library's canonical row-major layouts (the
// transposes are skipped when they are the identity: NHWC activations);
// 1-D convolutions get a unit H dimension. The workspace size is PlanConv's
// for the call, computed here (the handler re-plans and checks it).
//
// Two kinds, told apart by the dimension numbers JAX emits:
//  - "fwd": input [N, H, W, C] and weight [O, kH, kW, C] (the output
//    features first, the input features last) -> [N, oH, oW, O]. That
//    covers the forward convolution and JAX's input gradient, whose kernel
//    is kReverse(w) over its spatial dimensions: the reverse is folded into
//    flip = true when the convolution is its only user and it reverses
//    exactly the kernel's spatial dimensions.
//  - "wgrad": JAX's weight gradient (the transpose of the rhs), whose
//    lhs is the forward input with its batch and feature dimensions
//    swapped (the lhs feature dimension comes first) and whose rhs is the
//    output gradient (its input-feature, i.e. batch, dimension before its
//    output-feature one). The call gets the forward convolution's
//    parameters (stride = the rhs dilation, kernel dilation = the window
//    stride) and computes dW as patches x dY (conv.h kWeightGrad). With
//    only one of the two orders present the convolution is ambiguous and
//    left alone.
// A window reversed on every spatial dimension (XLA's algebraic simplifier
// swaps the operands of a convolution whose kernel is larger than its
// input, e.g. the weight gradient of an input-dilated one, and reverses the
// new kernel) toggles flip ("fwd") or reverses the rhs first ("wgrad").
//
// Left to the loop emitter: 3-D and higher, grouped (feature or batch
// groups), a partial window reversal, element types other than f32/f16/bf16 or a
// result type different from the operands', anything PlanConv refuses
// (negative low padding, 32-bit overflow), and convolutions of fewer than
// `min_flops` flops (2 * N * oH * oW * O * kH * kW * C), where a custom call
// costs more than the loop emitter's single fused kernel.
class MetalConvRewriter : public HloModulePass {
 public:
  // Measured on an M3 (bursts of 30 forward convolutions + relu, p10;
  // docs/performance.md): up to ~2.5 Mflop the loop emitter was 10-45%
  // faster (3.6 Mflop with C = 1 too: 0.084 vs 0.121 ms), from ~4.7 Mflop
  // metal$conv was (0.038 vs 0.046 ms, 1.6x at 19 Mflop).
  static constexpr int64_t kDefaultMinFlops = 4'000'000;

  explicit MetalConvRewriter(int64_t min_flops = kDefaultMinFlops)
      : min_flops_(min_flops) {}

  absl::string_view name() const override { return "metal-conv-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t min_flops_;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_CONV_REWRITER_H_
