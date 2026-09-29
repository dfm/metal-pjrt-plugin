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
//
// The weight gradient (ConvKind::kWeightGrad, MLX's vjp for the weight) is
// its own path, kWeightGrad: for the forward convolution described by the
// params, dW [O, kH * kW * C] = dY^T [O, N * oH * oW] x patches [N * oH *
// oW, kH * kW * C], the patches unfolded into the workspace in row chunks.
// That contraction is long and the product small, and steel has no split-K:
// each chunk's rows are split into `splits` equal parts, one batched steel
// GEMM accumulates the parts into f32 partial products in the workspace,
// and sum_splits adds them into dW (with one part and one chunk the GEMM
// writes dW directly).
//
// Every launch is charged its flops (rt::Stream::Launch) and none exceeds
// max_launch_flops: implicit and general split their row tiles over several
// launches (tile_m_offset), the explicit and weight-gradient paths bound
// their unfold chunks; so a large convolution spreads over several command
// buffers, each far from the GPU watchdog, like any other work.
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

// kForward: RunConv(in, weight -> out). kWeightGrad: RunConv(in, dY -> dW),
// the weight gradient of the forward convolution the params describe: dY
// has the output's shape [N, oH, oW, O], dW the weight's [O, kH, kW, C].
enum class ConvKind { kForward, kWeightGrad };

struct ConvParams {
  ConvKind kind = ConvKind::kForward;
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

enum class ConvPath {
  kImplicit,
  kPadChannels,
  kGeneral,
  kExplicit,
  kWeightGrad
};
const char* ConvPathName(ConvPath path);

struct ConvPlan {
  ConvPath path = ConvPath::kImplicit;
  ConvTile tile;             // kImplicit / kPadChannels / kGeneral
  int n_channels = 0;        // kImplicit: C when C <= 4, else 0
  bool small_filter = false; // kImplicit: kH, kW <= 16
  bool align_c = false;      // kGeneral: C % tile.bk == 0
  int64_t padded_c = 0;      // kPadChannels
  int64_t unfold_rows = 0;   // kExplicit, kWeightGrad: output pixels per chunk
  int64_t splits = 0;        // kWeightGrad: parts per chunk (f32 partials)
  int64_t split_rows = 0;    // kWeightGrad: rows per part
  int64_t tiles_m = 0;         // kImplicit..kGeneral: row tiles
  int64_t launch_tiles_m = 0;  // kImplicit..kGeneral: row tiles per launch
  uint64_t workspace_bytes = 0;
};

// The default max_launch_flops: half the command buffer's flop budget.
inline constexpr uint64_t kMaxLaunchFlops =
    rt::Stream::kMaxFlopsPerCommandBuffer / 2;

// The plan for `p`, or for `p` on path `*force` (tests; kWeightGrad is
// the only path of a kWeightGrad `p`). Unimplemented for what the kernels do
// not cover (groups > 1, negative pad_lo, an empty contraction C * kH * kW
// == 0, sizes past 32-bit indexing, a forced path that cannot run `p`);
// InvalidArgument for inconsistent sizes. `max_launch_flops` bounds each
// launch (tests pass small values to force the splits).
absl::StatusOr<ConvPlan> PlanConv(const ConvParams& p,
                                  const ConvPath* force = nullptr,
                                  uint64_t max_launch_flops = kMaxLaunchFlops);

// 2 * N * oH * oW * O * kH * kW * C / groups.
uint64_t ConvFlops(const ConvParams& p);

// Enqueues the convolution on the stream: `plan` is PlanConv(p) (or a
// forced plan), `workspace` at least plan.workspace_bytes (unused when 0).
// kForward: `b` is the weight, `out` the output; kWeightGrad: `b` is dY,
// `out` dW. Nothing is launched for an empty output; an empty contraction
// of the weight gradient (N * oH * oW == 0) zero-fills dW.
absl::Status RunConv(rt::Device* device, rt::Stream* stream,
                     const ConvParams& p, const ConvPlan& plan,
                     const void* in, const void* b, void* out,
                     void* workspace);

}  // namespace conv
}  // namespace metal_pjrt

#endif  // METAL_PJRT_CONV_CONV_H_
