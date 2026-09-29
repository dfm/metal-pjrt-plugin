// 2-D convolutions on the GPU with MLX's steel kernels (conv_kernels.h), no
// XLA, so conv_test runs them against rt::Device alone.
//
// Canonical layouts, row-major and dense: input [N, H, W, C], weight
// [O, kH, kW, C] (OHWI), output [N, oH, oW, O]. out[n, oh, ow, o] =
// sum over (kh, kw, c) of in_d[n, oh * stride - pad_lo + kh * kdil, ...] *
// w[o, kh', kw', c], where in_d is the input dilated by idil (idil - 1 zeros
// between elements) and zero outside, and kh' = flip ? kH - 1 - kh : kh
// (likewise kw'). The high padding is implied by the output size. f16/bf16
// accumulate in f32 (rounded once, when stored).
//
// PlanConv picks one of MLX's paths (conv.cpp dispatch_conv_2D_gpu, without
// Winograd and grouped convolutions for now):
//   kImplicit    specialized implicit GEMM: no input dilation, C <= 4 or
//                C % 16 == 0, and O <= 16 or O a multiple of the column tile
//                (MLX also takes other O % 16 == 0, reading weight rows past
//                the end of the buffer; those go to kGeneral here);
//   kPadChannels the input and weight copied with C zero-padded to a
//                multiple of 16 into the workspace, then kImplicit (stride 1,
//                no dilation of the input, >= 256 output pixels, >= 9 taps);
//   kGeneral     general implicit GEMM: input dilation, any C and O (C and O
//                multiples of 16, or >= 256 output pixels);
//   kExplicit    unfold (im2col) row tiles into the workspace + steel GEMM.
#ifndef METAL_PJRT_CONV_CONV_H_
#define METAL_PJRT_CONV_CONV_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/conv/conv_kernels.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace conv {

struct ConvParams {
  ConvType type = ConvType::kF32;
  int64_t n = 0, h = 0, w = 0, c = 0;  // input
  int64_t o = 0, kh = 0, kw = 0;       // weight [o, kh, kw, c]
  int64_t out_h = 0, out_w = 0;        // output [n, out_h, out_w, o]
  int64_t stride[2] = {1, 1};
  int64_t pad_lo[2] = {0, 0};
  int64_t kdil[2] = {1, 1};  // kernel (rhs) dilation
  int64_t idil[2] = {1, 1};  // input (lhs) dilation
  int64_t groups = 1;
  bool flip = false;
};

std::string ConvParamsDebugString(const ConvParams& p);

enum class ConvPath { kImplicit, kPadChannels, kGeneral, kExplicit };
const char* ConvPathName(ConvPath path);

struct ConvPlan {
  ConvPath path = ConvPath::kImplicit;
  ConvTile tile;             // kImplicit / kPadChannels / kGeneral
  int n_channels = 0;        // kImplicit: C when C <= 4, else 0
  bool small_filter = false; // kImplicit: kH, kW <= 16
  bool align_c = false;      // kGeneral: C % tile.bk == 0
  int64_t padded_c = 0;      // kPadChannels
  int64_t unfold_rows = 0;   // kExplicit: output pixels per unfold tile
  uint64_t workspace_bytes = 0;
};

// The plan for `p`, or for `p` on path `*force` (tests). Unimplemented for
// what the kernels do not cover (groups > 1, negative pad_lo, an empty
// contraction C * kH * kW == 0, sizes past 32-bit indexing, a forced path
// that cannot run `p`); InvalidArgument for inconsistent sizes.
absl::StatusOr<ConvPlan> PlanConv(const ConvParams& p,
                                  const ConvPath* force = nullptr);

// 2 * N * oH * oW * O * kH * kW * C / groups.
uint64_t ConvFlops(const ConvParams& p);

// Enqueues the convolution on the stream: `plan` is PlanConv(p) (or a
// forced plan), `workspace` at least plan.workspace_bytes (unused when 0).
// Nothing is launched for an empty output.
absl::Status RunConv(rt::Device* device, rt::Stream* stream,
                     const ConvParams& p, const ConvPlan& plan,
                     const void* in, const void* weight, void* out,
                     void* workspace);

}  // namespace conv
}  // namespace metal_pjrt

#endif  // METAL_PJRT_CONV_CONV_H_
