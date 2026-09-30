// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// MSL source for the "steel" implicit-GEMM 2-D convolutions (conv/conv.h).
// A port of MLX's steel convolution kernels (mlx/backend/metal/kernels/
// steel/conv/{params.h,loaders/*.h,kernels/steel_conv.h,
// kernels/steel_conv_general.h}, MLX 0.32.2), which are:
//
//   Copyright (c) 2023-2024 Apple Inc. MIT License (MLX).
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions: The above copyright notice and this permission
//   notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
//
// Changes from MLX: self-contained (no MLX headers; the simdgroup MMA is the
// one of kernels/steel_gemm.metal without its epilogue, in namespace
// steel_conv), 2-D only, the general kernel's align_C switch is function
// constant 0, one instantiation per compiled source (see the end), and the
// large-filter input loader's bounds check follows flip (see there).
//
// Layouts (row-major): input [N, iH, iW, C], weight [O, kH, kW, C / groups],
// output [N, oH, oW, O]. The GEMM is M = N * oH * oW rows, N = O / groups
// columns, K = kH * kW * C / groups, accumulated in f32 for every T.
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// implicit_gemm_conv_2d_general only: C % BK == 0 (ConvKernelConstants).
constant bool ALIGN_C [[function_constant(0)]];

// Must match SteelConvParams in conv/conv_kernels.cc (MLXConvParams<2>).
struct SteelConvParams {
  int N;        // batch
  int C;        // input channels
  int O;        // output channels
  int iS[2];    // input spatial size
  int wS[2];    // kernel spatial size
  int oS[2];    // output spatial size
  int str[2];   // strides
  int pad[2];   // low padding (the high side is implied by oS)
  int kdil[2];  // kernel (rhs) dilation
  int idil[2];  // input (lhs) dilation
  long in_strides[4];
  long wt_strides[4];
  long out_strides[4];
  int groups;
  bool flip;  // correlate with the spatially reversed kernel
};

// Must match ImplicitGemmParams in conv/conv_kernels.cc.
struct ImplicitGemmConv2DParams {
  int M;
  int N;
  int K;
  int gemm_k_iterations;
  int inp_jump_w;
  int inp_jump_h;
  int inp_jump_c;
  int tiles_n;
  int tiles_m;
  int swizzle_log;
  // Row tiles [tile_m_offset, tile_m_offset + grid height) of tiles_m: a
  // large convolution is split into several launches (conv/conv.cc).
  int tile_m_offset;
};

// Must match GeneralJumpParams in conv/conv_kernels.cc.
struct Conv2DGeneralJumpParams {
  int f_wgt_jump_h;
  int f_wgt_jump_w;
  int f_out_jump_h;
  int f_out_jump_w;
  int adj_out_h;
  int adj_out_w;
  int adj_out_hw;
  int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

namespace steel_conv {

// Per-simdgroup MMA over a (BM, BK) x (BK, BN) threadgroup tile with f32
// accumulators in 8x8 simdgroup matrices (MLX BlockMMA; the same as
// kernels/steel_gemm.metal's). Each lane owns two adjacent columns of one
// row of every 8x8 fragment.
template <typename Ty, int BM_, int BN_, int BK_, int WM_, int WN_, bool TA,
          bool TB, short lda_tgp, short ldb_tgp>
struct BlockMMA {
  typedef simdgroup_matrix<float, 8, 8> mat;
  STEEL_CONST short kFrag = 8;
  STEEL_CONST short TM_stride = kFrag * WM_;
  STEEL_CONST short TN_stride = kFrag * WN_;
  STEEL_CONST short TM = BM_ / TM_stride;
  STEEL_CONST short TN = BN_ / TN_stride;
  STEEL_CONST short A_str_m = TA ? 1 : lda_tgp;
  STEEL_CONST short A_str_k = TA ? lda_tgp : 1;
  STEEL_CONST short B_str_k = TB ? 1 : ldb_tgp;
  STEEL_CONST short B_str_n = TB ? ldb_tgp : 1;
  STEEL_CONST short tile_stride_a = kFrag * A_str_k;
  STEEL_CONST short tile_stride_b = kFrag * B_str_k;

  mat Atile[TM];
  mat Btile[TN];
  mat Ctile[TM * TN];
  short sm, sn;
  short As_offset, Bs_offset;

  METAL_FUNC BlockMMA(ushort simd_group_id, ushort simd_lane_id) thread {
    short tm = kFrag * (simd_group_id / WN_);
    short tn = kFrag * (simd_group_id % WN_);
    const short qid = simd_lane_id / 4;
    sm = (qid & 4) + ((simd_lane_id / 2) % 4);
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    As_offset = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_offset = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm;
    sn += tn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM * TN; i++)
      Ctile[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  METAL_FUNC void mma(const threadgroup Ty* As, const threadgroup Ty* Bs)
      thread {
    As += As_offset;
    Bs += Bs_offset;
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK_; kk += kFrag) {
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        const threadgroup Ty* p = As + i * TM_stride * A_str_m;
        Atile[i].thread_elements()[0] = float(p[0]);
        Atile[i].thread_elements()[1] = float(p[A_str_k]);
      }
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        const threadgroup Ty* p = Bs + j * TN_stride * B_str_n;
        Btile[j].thread_elements()[0] = float(p[0]);
        Btile[j].thread_elements()[1] = float(p[B_str_n]);
      }
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; n++) {
          short ns = (m % 2) ? (TN - 1 - n) : n;  // serpentine
          simdgroup_multiply_accumulate(Ctile[m * TN + ns], Atile[m],
                                        Btile[ns], Ctile[m * TN + ns]);
        }
      }
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  // Edge-safe store of the tile: dims = (cols, rows) valid from its origin.
  METAL_FUNC void store_result_safe(device Ty* D, const int ldd,
                                    short2 dims) const thread {
    D += sm * ldd + sn;
    dims -= short2(sn, sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      if (i * TM_stride < dims.y) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          thread const auto& acc = Ctile[i * TN + j].thread_elements();
          const int off = (i * TM_stride) * ldd + j * TN_stride;
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < 2; k++) {
            if (j * TN_stride + k < dims.x) D[off + k] = Ty(acc[k]);
          }
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Loaders of the specialized kernel (MLX loaders/loader_channel_{n,l}.h).
// ---------------------------------------------------------------------------

template <short n_channels_>
struct ChannelHelper {
  STEEL_CONST short n_channels = n_channels_;
  STEEL_CONST short vec_size = n_channels_ <= 4 ? 4 : 8;
  STEEL_CONST short excess = vec_size - n_channels_;
};

template <>
struct ChannelHelper<1> {
  STEEL_CONST short n_channels = 1;
  STEEL_CONST short vec_size = 1;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<2> {
  STEEL_CONST short n_channels = 2;
  STEEL_CONST short vec_size = 2;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<3> {
  STEEL_CONST short n_channels = 3;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 1;
};

template <>
struct ChannelHelper<4> {
  STEEL_CONST short n_channels = 4;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 0;
};

// Input tile for C <= 4: each K step covers BK / vec_size filter taps with
// the channels zero-padded to vec_size.
template <typename T, short BM, short BN, short BK, short tgp_size,
          short n_channels, short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallChannels {
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant SteelConvParams* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  int weight_hw;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  METAL_FUNC Conv2DInputBlockLoaderSmallChannels(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id, uint simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_hw(thread_idx % TCOLS) {
    int out_n_pixels = params->oS[0] * params->oS[1];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
               iw * params->in_strides[2];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
    }
  }

  METAL_FUNC void load_unsafe() const thread {
    if (weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
      }
      return;
    }
    int wh = (weight_hw / params->wS[1]);
    int ww = (weight_hw % params->wS[1]);
    int flip_h = params->flip ? params->wS[0] - wh - 1 : wh;
    int flip_w = params->flip ? params->wS[1] - ww - 1 : ww;
    int weight_h = flip_h * params->kdil[0];
    int weight_w = flip_w * params->kdil[1];
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int ih = read_ih[i] + weight_h;
      int iw = read_iw[i] + weight_w;
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        const device T* curr_src = src[i] + weight_h * params->in_strides[1] +
                                   weight_w * params->in_strides[2];
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; ++j)
          dst[is * dst_ld + j] = curr_src[j];
        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; ++j)
          dst[is * dst_ld + j] = T(0);
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void next() thread { weight_hw += TCOLS; }
};

// Weight tile for C <= 4 (matches Conv2DInputBlockLoaderSmallChannels).
template <typename T, short BM, short BN, short BK, short tgp_size,
          short n_channels, short tgp_padding = 0>
struct Conv2DWeightBlockLoaderSmallChannels {
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  const constant SteelConvParams* params;
  int weight_hw;
  const int read_n;
  const bool do_read;

  METAL_FUNC Conv2DWeightBlockLoaderSmallChannels(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id, uint simd_lane_id) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld),
        params(params_),
        weight_hw(thread_idx % TCOLS),
        read_n(offsets.y + bi),
        do_read(read_n + BN <= gemm_params_->N) {}

  METAL_FUNC void load_unsafe() const thread {
    if (bi >= BROWS || bj >= BCOLS) return;
    if (read_n >= params->O || weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
      }
      return;
    }
    const device T* curr_src = src + weight_hw * (params->C / params->groups);
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; j++)
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; j++)
          dst[i * dst_ld + j] = T(0);
      }
    } else {
      for (short i = 0; i < BROWS; i += TROWS) {
        if (((read_n + i) < params->O)) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < n_channels; j++)
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          STEEL_PRAGMA_UNROLL
          for (short j = n_channels; j < vec_size; j++)
            dst[i * dst_ld + j] = T(0);
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
        }
      }
    }
  }

  METAL_FUNC void next() thread { weight_hw += TCOLS; }
};

// Input tile, any filter size: bounds checked per tap.
template <typename T, short BM, short BN, short BK, short tgp_size,
          short tgp_padding = 0>
struct Conv2DInputBlockLoaderLargeFilter {
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant SteelConvParams* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  METAL_FUNC Conv2DInputBlockLoaderLargeFilter(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id, uint simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
               iw * params->in_strides[2] + bj;
    }
  }

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      // With flip, src walks the reversed taps (inp_jump_* negated), so the
      // bounds check must use the reversed tap too. (MLX checks the
      // unreversed one: wrong for flip with a kernel over 16 taps wide.)
      int h = params->flip ? params->wS[0] - 1 - weight_h : weight_h;
      int w = params->flip ? params->wS[1] - 1 - weight_w : weight_w;
      int ih = read_ih[i] + h * params->kdil[0];
      int iw = read_iw[i] + w * params->kdil[1];
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = src[i][j];
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_w;
      return;
    }
    weight_w = 0;
    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_h;
      return;
    }
    weight_h = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_c;
  }
};

// Input tile for kH, kW <= 16: the in-bounds taps precomputed as bit masks.
template <typename T, short BM, short BN, short BK, short tgp_size,
          short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallFilter {
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;
  using mask_t = short;

  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant SteelConvParams* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  METAL_FUNC Conv2DInputBlockLoaderSmallFilter(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id, uint simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];
    int read_n[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
               iw * params->in_strides[2] + bj;
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_h[i] = 0;
      mask_w[i] = 0;
    }
    for (short kh = 0; kh < params->wS[0]; kh++) {
      short flip_h = params->flip ? params->wS[0] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int ih = read_ih[i] + flip_h * params->kdil[0];
        bool in_bounds = n < params->N && ih >= 0 && ih < params->iS[0];
        mask_h[i] |= (in_bounds << kh);
      }
    }
    for (short kw = 0; kw < params->wS[1]; kw++) {
      short flip_w = params->flip ? params->wS[1] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[1];
        bool in_bounds = iw >= 0 && iw < params->iS[1];
        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  METAL_FUNC void load_unsafe() const thread {
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      if ((mask_h[i] & h_mask) && (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = src[i][j];
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_w;
      return;
    }
    weight_w = 0;
    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_h;
      return;
    }
    weight_h = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) src[i] += gemm_params->inp_jump_c;
  }
};

// Weight tile of the specialized kernel for C % BK == 0.
template <typename T, short BM, short BN, short BK, short tgp_size,
          short tgp_padding = 0>
struct Conv2DWeightBlockLoader {
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  const constant SteelConvParams* params;
  int weight_hw;
  int weight_step;
  const int read_n;
  const bool do_read;

  METAL_FUNC Conv2DWeightBlockLoader(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id, uint simd_lane_id) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_hw(0),
        weight_step(params_->C / params_->groups),
        read_n(offsets.y + bi),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  METAL_FUNC void load_unsafe() const thread {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = src[i * src_ld + j];
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++)
            dst[i * dst_ld + j] = src[i * src_ld + j];
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
        }
      }
    }
  }

  METAL_FUNC void next() thread {
    if (++weight_hw < (params->wS[1] * params->wS[0])) {
      src += weight_step;
      return;
    }
    weight_hw = 0;
    src += BK - (params->wS[1] * params->wS[0] - 1) * weight_step;
  }
};

// ---------------------------------------------------------------------------
// Loaders of the general kernel (MLX loaders/loader_general.h): input
// dilation, any channel count (a masked last K step when !ALIGN_C).
// ---------------------------------------------------------------------------

template <typename T, short BM, short BN, short BK, short tgp_size,
          short tgp_padding = 0>
struct Conv2DInputBlockLoaderGeneral {
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant SteelConvParams* params;
  const constant Conv2DGeneralJumpParams* jump_params;
  const short base_wh;
  const short base_ww;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  METAL_FUNC Conv2DInputBlockLoaderGeneral(
      const device T* src_, threadgroup T* dst_, const int4 offsets,
      const constant SteelConvParams* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_, const short base_ww_, uint simd_group_id,
      uint simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / jump_params->adj_out_hw;
      int hw = offset_nhw % jump_params->adj_out_hw;
      int oh = (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h +
               offsets.z;
      int ow = (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w +
               offsets.w;
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
      src[i] = src_ + n * params->in_strides[0] + bj;
    }
  }

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;
      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];
      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];
      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j)
          dst[is * dst_ld + j] = (src[i])[offset + j];
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void load_safe(const short remaining_k) const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;
      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];
      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];
      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
        if (bj + vec_size <= remaining_k) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; ++j)
            dst[is * dst_ld + j] = (src[i])[offset + j];
        } else {
          for (short j = 0; j < vec_size; ++j) {
            dst[is * dst_ld + j] =
                bj + j < remaining_k ? (src[i])[offset + j] : T(0);
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) dst[is * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) return;
    weight_w = base_ww;
    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) return;
    weight_h = base_wh;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) src[i] += BK;
  }
};

template <typename T, short BM, short BN, short BK, short tgp_size,
          short tgp_padding = 0>
struct Conv2DWeightBlockLoaderGeneral {
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  STEEL_CONST short n_rows = BROWS / TROWS;

  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  const constant SteelConvParams* params;
  const constant Conv2DGeneralJumpParams* jump_params;
  const short base_wh;
  const short base_ww;
  short weight_h;
  short weight_w;
  const int start_row;

  METAL_FUNC Conv2DWeightBlockLoaderGeneral(
      const device T* src_, threadgroup T* dst_, const int2 offsets,
      const constant SteelConvParams* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_, const short base_ww_, uint simd_group_id,
      uint simd_lane_id) thread
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_),
        start_row(offsets.y + bi) {}

  METAL_FUNC void load_unsafe() const thread {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
                               weight_w * params->wt_strides[2];
    if ((start_row + BN <= params->O)) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((start_row + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++)
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
        }
      }
    }
  }

  METAL_FUNC void load_safe(const short remaining_k) const thread {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
                               weight_w * params->wt_strides[2];
    for (short i = 0; i < BN; i += TROWS) {
      if ((start_row + i) < params->O) {
        if (bj + vec_size <= remaining_k) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++)
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        } else {
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] =
                bj + j < remaining_k ? curr_src[i * src_ld + j] : T(0);
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = T(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) return;
    weight_w = base_ww;
    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) return;
    weight_h = base_wh;
    src += BK;
  }
};

}  // namespace steel_conv

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

// Specialized implicit GEMM (MLX steel_conv.h): no input dilation; C <= 4
// (N_CHANNELS = C) or C % 16 == 0 (N_CHANNELS = 0); O <= 16 (BN = 8) or
// O % BN == 0. SMALL_FILTER: kH, kW <= 16 (masked taps). Grid (tiles_n, a
// range of tiles_m, groups), WM * WN simdgroups.
template <typename T, int BM, int BN, int BK, int WM, int WN, int N_CHANNELS,
          bool SMALL_FILTER>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
implicit_gemm_conv_2d(const device T* A [[buffer(0)]],
                      const device T* B [[buffer(1)]],
                      device T* C [[buffer(2)]],
                      const constant SteelConvParams* params [[buffer(3)]],
                      const constant ImplicitGemmConv2DParams* gemm_params
                      [[buffer(4)]],
                      uint3 tid [[threadgroup_position_in_grid]],
                      uint simd_gid [[simdgroup_index_in_threadgroup]],
                      uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace steel_conv;
  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);
  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;
  constexpr short tgp_size = WM * WN * 32;

  using loader_a_t = typename metal::conditional_t<
      N_CHANNELS != 0 && N_CHANNELS <= 4,
      Conv2DInputBlockLoaderSmallChannels<T, BM, BN, BK, tgp_size, N_CHANNELS,
                                          tgp_padding_a>,
      typename metal::conditional_t<
          SMALL_FILTER,
          Conv2DInputBlockLoaderSmallFilter<T, BM, BN, BK, tgp_size,
                                            tgp_padding_a>,
          Conv2DInputBlockLoaderLargeFilter<T, BM, BN, BK, tgp_size,
                                            tgp_padding_a>>>;
  using loader_b_t = typename metal::conditional_t<
      N_CHANNELS != 0 && N_CHANNELS <= 4,
      Conv2DWeightBlockLoaderSmallChannels<T, BM, BN, BK, tgp_size,
                                           N_CHANNELS, tgp_padding_b>,
      Conv2DWeightBlockLoader<T, BM, BN, BK, tgp_size, tgp_padding_b>>;
  using mma_t = BlockMMA<T, BM, BN, BK, WM, WN, transpose_a, transpose_b,
                         shape_a_cols, shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
                    ((tid.x) & ((1 << gemm_params->swizzle_log) - 1)) +
                    gemm_params->tile_m_offset;
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;
  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) return;

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  const int N = gemm_params->N;
  const int C_per_group = params->C / params->groups;

  A += tid.z * C_per_group;
  B += tid.z * N * K;
  C += tid.z * N;
  B += c_col * K;
  C += static_cast<size_t>(c_row) * N * params->groups + c_col;

  const int2 offsets_a(0, c_row);
  const int2 offsets_b(0, c_col);
  loader_a_t loader_a(A, As, offsets_a, params, gemm_params, simd_gid,
                      simd_lid);
  loader_b_t loader_b(B, Bs, offsets_b, params, gemm_params, simd_gid,
                      simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  int gemm_k_iterations = gemm_params->gemm_k_iterations;
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  threadgroup_barrier(mem_flags::mem_none);

  short tgp_bm = min(BM, gemm_params->M - c_row);
  short tgp_bn = min(BN, gemm_params->N - c_col);
  const int ldc = N * params->groups;
  mma_op.store_result_safe(C, ldc, short2(tgp_bn, tgp_bm));
}

// General implicit GEMM (MLX steel_conv_general.h): input dilation, any C
// (ALIGN_C: C % BK == 0) and O. The output is computed in f_out_jump_h *
// f_out_jump_w interleaved phases (grid z), each with the taps that land on
// real (undilated) input rows/columns, from base_h / base_w.
template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
implicit_gemm_conv_2d_general(
    const device T* A [[buffer(0)]], const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant SteelConvParams* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    const constant Conv2DGeneralJumpParams* jump_params [[buffer(5)]],
    const constant Conv2DGeneralBaseInfo* base_h [[buffer(6)]],
    const constant Conv2DGeneralBaseInfo* base_w [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace steel_conv;
  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);
  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;
  constexpr short tgp_size = WM * WN * 32;

  using loader_a_t =
      Conv2DInputBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_a>;
  using loader_b_t =
      Conv2DWeightBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_b>;
  using mma_t = BlockMMA<T, BM, BN, BK, WM, WN, transpose_a, transpose_b,
                         shape_a_cols, shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
                    ((tid.x) & ((1 << gemm_params->swizzle_log) - 1)) +
                    gemm_params->tile_m_offset;
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;
  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) return;

  const int tid_z = tid.z;
  const int base_oh = tid_z / jump_params->f_out_jump_w;
  const int base_ow = tid_z % jump_params->f_out_jump_w;
  const int base_wh = base_h[base_oh].weight_base;
  const int base_ww = base_w[base_ow].weight_base;
  const int base_wh_size = base_h[base_oh].weight_size;
  const int base_ww_size = base_w[base_ow].weight_size;

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  B += c_col * K;

  const int4 offsets_a(0, c_row, base_oh, base_ow);
  const int2 offsets_b(0, c_col);
  loader_a_t loader_a(A, As, offsets_a, params, jump_params, base_wh, base_ww,
                      simd_gid, simd_lid);
  loader_b_t loader_b(B, Bs, offsets_b, params, jump_params, base_wh, base_ww,
                      simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (ALIGN_C) {
    int gemm_k_iterations =
        base_wh_size * base_ww_size * gemm_params->gemm_k_iterations;
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  } else {
    for (int k = 1; k < gemm_params->gemm_k_iterations; k++) {
      for (int j = 0; j < base_wh_size * base_ww_size; j++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_unsafe();
        loader_b.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
        loader_a.next();
        loader_b.next();
      }
    }
    const short remaining_k = params->C % BK;
    for (int j = 0; j < base_wh_size * base_ww_size; j++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(remaining_k);
      loader_b.load_safe(remaining_k);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  // Rows of this phase are scattered over the output: store per row.
  int offset_m = c_row + mma_op.sm;
  int offset_n = c_col + mma_op.sn;
  C += offset_n;
  if (offset_n >= gemm_params->N) return;
  short diff = gemm_params->N - offset_n;
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < mma_t::TM; i++) {
    int cm = offset_m + i * mma_t::TM_stride;
    int n = cm / jump_params->adj_out_hw;
    int hw = cm % jump_params->adj_out_hw;
    int oh = (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h +
             base_oh;
    int ow = (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w +
             base_ow;
    if (n < params->N && oh < params->oS[0] && ow < params->oS[1]) {
      size_t offset_cm = static_cast<size_t>(n) * params->out_strides[0] +
                         oh * params->out_strides[1] +
                         ow * params->out_strides[2];
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < mma_t::TN; j++) {
        thread const auto& accum =
            mma_op.Ctile[i * mma_t::TN + j].thread_elements();
        size_t offset = offset_cm + (j * mma_t::TN_stride);
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < 2; k++) {
          if ((j * mma_t::TN_stride + k) < diff) C[offset + k] = T(accum[k]);
        }
      }
    }
  }
}

// Each source compiles one variant: conv_kernels.cc appends one of
//   instantiate_implicit_gemm_conv_2d("host name", T, BM, BN, BK, WM, WN,
//                                     N_CHANNELS, SMALL_FILTER)
//   instantiate_implicit_gemm_conv_2d_general("host name", T, BM, BN, BK, WM,
//                                             WN)
#define instantiate_implicit_gemm_conv_2d(name, T, BM, BN, BK, WM, WN, NC, SF) \
  template [[host_name(name)]] [[kernel]]                                     \
  decltype(implicit_gemm_conv_2d<T, BM, BN, BK, WM, WN, NC, SF>)              \
      implicit_gemm_conv_2d<T, BM, BN, BK, WM, WN, NC, SF>;
#define instantiate_implicit_gemm_conv_2d_general(name, T, BM, BN, BK, WM, WN) \
  template [[host_name(name)]] [[kernel]]                                      \
  decltype(implicit_gemm_conv_2d_general<T, BM, BN, BK, WM, WN>)               \
      implicit_gemm_conv_2d_general<T, BM, BN, BK, WM, WN>;
