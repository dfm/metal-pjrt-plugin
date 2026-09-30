// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/conv.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.
//
// The convolution kernels (kernels/steel_conv.metal, kernels/conv_misc.metal)
// as the host sees them: the parameter structs they take, the MSL source and
// function name of each variant, and the tile rules that pick one. No XLA;
// the dispatch is conv.h.
#ifndef METAL_PJRT_CONV_CONV_KERNELS_H_
#define METAL_PJRT_CONV_CONV_KERNELS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace conv {

enum class ConvType { kF32, kF16, kBF16 };

// "float" | "half" | "bfloat".
const char* MslTypeName(ConvType t);

// Must match SteelConvParams in the .metal files (MLX's MLXConvParams<2>).
// Row-major NHWC input, OHWI weight (I = C / groups), NHWC output; the
// strides are in elements.
struct SteelConvParams {
  int32_t N, C, O;
  int32_t iS[2], wS[2], oS[2];
  int32_t str[2], pad[2], kdil[2], idil[2];
  int64_t in_strides[4], wt_strides[4], out_strides[4];
  int32_t groups;
  bool flip;
};
static_assert(sizeof(SteelConvParams) == 176, "layout must match the MSL");

// Must match ImplicitGemmConv2DParams in steel_conv.metal.
struct ImplicitGemmParams {
  int32_t M, N, K;
  int32_t gemm_k_iterations;
  int32_t inp_jump_w, inp_jump_h, inp_jump_c;
  int32_t tiles_n, tiles_m;
  int32_t swizzle_log;
  int32_t tile_m_offset;  // first row tile of this launch
};
static_assert(sizeof(ImplicitGemmParams) == 44, "layout must match the MSL");

// Must match Conv2DGeneralJumpParams in steel_conv.metal.
struct GeneralJumpParams {
  int32_t f_wgt_jump_h, f_wgt_jump_w;
  int32_t f_out_jump_h, f_out_jump_w;
  int32_t adj_out_h, adj_out_w, adj_out_hw, adj_implicit_m;
};
static_assert(sizeof(GeneralJumpParams) == 32, "layout must match the MSL");

// Must match Conv2DGeneralBaseInfo in steel_conv.metal.
struct GeneralBaseInfo {
  int32_t weight_base, weight_size;
};

// Must match UnfoldRows / PadRows / SumSplits in conv_misc.metal.
struct UnfoldRows {
  int32_t row_offset, rows, vecs;
};
struct PadRows {
  uint32_t rows, cols, out_cols;
};
struct SumSplits {
  uint32_t count, splits;
};

struct ConvTile {
  int bm = 32, bn = 32, bk = 16, wm = 2, wn = 2;
  bool operator==(const ConvTile& o) const {
    return bm == o.bm && bn == o.bn && bk == o.bk && wm == o.wm && wn == o.wn;
  }
};

// MLX's tile rules (conv.cpp implicit_gemm_conv_2D_gpu and
// implicit_gemm_conv_2D_general_gpu). `m`: GEMM rows (N * oH * oW; for the
// general kernel, the rows of one output phase); `n`: output channels per
// group; `c`: input channels per group.
ConvTile ImplicitTile(int64_t m, int64_t n, int64_t c);
ConvTile GeneralTile(int64_t m, int64_t n, int64_t c);

// One compilable variant: the MSL (for steel, kernels/steel_conv.metal plus
// one explicit instantiation), the function and its function constants.
struct ConvKernelSource {
  std::string msl;
  std::string function;
  std::vector<rt::FunctionConstant> constants;
};

// implicit_gemm_conv_2d: n_channels = C for C <= 4, else 0; small_filter =
// kH, kW <= 16 (only when n_channels == 0).
ConvKernelSource ImplicitConvKernel(ConvType t, const ConvTile& tile,
                                    int n_channels, bool small_filter);
// implicit_gemm_conv_2d_general; align_c = C % bk == 0.
ConvKernelSource GeneralConvKernel(ConvType t, const ConvTile& tile,
                                   bool align_c);
// unfold_2d_<vector_bytes> (2, 4, 8 or 16; a bit copy, so any type) /
// pad_cols_<type> / sum_splits_<type> of conv_misc.metal.
ConvKernelSource UnfoldKernel(int vector_bytes);
ConvKernelSource PadColsKernel(ConvType t);
ConvKernelSource SumSplitsKernel(ConvType t);

// Every variant the dispatch (conv.h) can select, each once: the tile rules
// applied across their branch points, for every type (kernels_test compiles
// them all).
std::vector<ConvKernelSource> AllConvKernels();

}  // namespace conv
}  // namespace metal_pjrt

#endif  // METAL_PJRT_CONV_CONV_KERNELS_H_
