// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/conv.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.

#include "metal_pjrt/conv/conv.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/runtime/kernel_launch.h"

namespace metal_pjrt {
namespace conv {
namespace {

// The explicit and weight-gradient paths unfold at most this much of the
// patch matrix at a time (whole rows; at least one).
constexpr uint64_t kUnfoldMaxBytes = 64ull << 20;

// Weight gradient split-K: enough threadgroups for the GPU (the M3 has 10
// cores; 512 beat 256 by 2-5% and 128 by up to 1.6x on the CIFAR layers),
// each part at least kMinSplitRows long, the f32 partial products at most
// kWgradMaxPartialsBytes (more parts only when they fit).
constexpr int64_t kWgradTargetGroups = 512;
constexpr int64_t kMinSplitRows = 256;
constexpr uint64_t kWgradMaxPartialsBytes = 8ull << 20;

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

// SteelGemmSupports' leading-dimension limit (steel_gemm.cc; conv_test
// checks the two agree): the explicit and weight-gradient GEMMs have
// leading dimensions kH * kW * C and O.
constexpr int64_t kSteelMaxLd = kInt32Max / 256;

int64_t ItemSize(ConvType t) { return t == ConvType::kF32 ? 4 : 2; }

blas::MpsDType GemmType(ConvType t) {
  return t == ConvType::kF32   ? blas::MpsDType::kF32
         : t == ConvType::kF16 ? blas::MpsDType::kF16
                               : blas::MpsDType::kBF16;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t OutPixels(const ConvParams& p) { return p.n * p.out_h * p.out_w; }

// The weight gradient's f32 partials follow the patches in the workspace
// (patch rows * k 16- or 32-bit elements), at a 256-byte boundary: an odd
// number of f16/bf16 elements would leave them 2-byte aligned.
uint64_t WgradPartialsOffset(int64_t unfold_rows, int64_t k, int64_t item) {
  const uint64_t bytes = static_cast<uint64_t>(unfold_rows * k * item);
  return (bytes + 255) / 256 * 256;
}

// The flops a path's kernels execute: kPadChannels runs over the padded
// channels.
uint64_t PlannedFlops(const ConvParams& p, const ConvPlan& plan) {
  if (plan.path != ConvPath::kPadChannels) return ConvFlops(p);
  ConvParams padded = p;
  padded.c = plan.padded_c;
  return ConvFlops(padded);
}

bool Stride1(const ConvParams& p) {
  return p.stride[0] == 1 && p.stride[1] == 1;
}
bool NoInputDilation(const ConvParams& p) {
  return p.idil[0] == 1 && p.idil[1] == 1;
}

// kImplicit's output-channel condition for input channels `c`: O <= 16 (the
// 8-wide tile checks rows) or a whole number of column tiles (the wider
// tiles' weight loader does not).
bool ImplicitOutputOk(const ConvParams& p, int64_t c) {
  return p.o <= 16 || p.o % ImplicitTile(OutPixels(p), p.o, c).bn == 0;
}

bool ImplicitChannelsOk(int64_t c) { return c <= 4 || c % 16 == 0; }

absl::Status Unsupported(const ConvParams& p, absl::string_view why) {
  return absl::UnimplementedError(
      absl::StrCat("convolution not supported on Metal (", why,
                   "): ", ConvParamsDebugString(p)));
}

SteelConvParams MakeSteelParams(const ConvParams& p, int64_t c) {
  SteelConvParams s = {};
  s.N = static_cast<int32_t>(p.n);
  s.C = static_cast<int32_t>(c);
  s.O = static_cast<int32_t>(p.o);
  s.iS[0] = static_cast<int32_t>(p.h);
  s.iS[1] = static_cast<int32_t>(p.w);
  s.wS[0] = static_cast<int32_t>(p.kh);
  s.wS[1] = static_cast<int32_t>(p.kw);
  s.oS[0] = static_cast<int32_t>(p.out_h);
  s.oS[1] = static_cast<int32_t>(p.out_w);
  for (int i = 0; i < 2; ++i) {
    s.str[i] = static_cast<int32_t>(p.stride[i]);
    s.pad[i] = static_cast<int32_t>(p.pad_lo[i]);
    s.kdil[i] = static_cast<int32_t>(p.kdil[i]);
    s.idil[i] = static_cast<int32_t>(p.idil[i]);
  }
  s.in_strides[0] = p.h * p.w * c;
  s.in_strides[1] = p.w * c;
  s.in_strides[2] = c;
  s.in_strides[3] = 1;
  s.wt_strides[0] = p.kh * p.kw * c;
  s.wt_strides[1] = p.kw * c;
  s.wt_strides[2] = c;
  s.wt_strides[3] = 1;
  s.out_strides[0] = p.out_h * p.out_w * p.o;
  s.out_strides[1] = p.out_w * p.o;
  s.out_strides[2] = p.o;
  s.out_strides[3] = 1;
  s.groups = static_cast<int32_t>(p.groups);
  s.flip = p.flip;
  return s;
}

rt::Dim3 Threads(const ConvTile& t) {
  return {static_cast<uint32_t>(32 * t.wm * t.wn), 1, 1};
}

// Launches `kernel` over the row tiles [0, tiles_m) in ranges of
// plan.launch_tiles_m (gp.tile_m_offset = the range's first tile), each
// charged its share of the flops. args[4] is gp (KernelArg::Bytes copies,
// so it is set per launch).
absl::Status LaunchRowTiles(rt::Stream* stream, const rt::Kernel& kernel,
                            const ConvParams& p, const ConvPlan& plan,
                            ImplicitGemmParams& gp, uint32_t grid_z,
                            std::vector<rt::KernelArg> args) {
  const int64_t tiles_m = gp.tiles_m;
  const uint64_t tile_flops =
      (PlannedFlops(p, plan) + tiles_m - 1) / std::max<int64_t>(tiles_m, 1);
  const int64_t step = plan.launch_tiles_m > 0 ? plan.launch_tiles_m : tiles_m;
  for (int64_t first = 0; first < tiles_m; first += step) {
    const int64_t tiles = std::min(step, tiles_m - first);
    gp.tile_m_offset = static_cast<int32_t>(first);
    args[4] = rt::KernelArg::Bytes(&gp, sizeof(gp));
    const rt::Dim3 groups{static_cast<uint32_t>(gp.tiles_n),
                          static_cast<uint32_t>(tiles), grid_z};
    ABSL_RETURN_IF_ERROR(stream->Launch(kernel, groups, Threads(plan.tile),
                                        args, /*threadgroup_memory_bytes=*/0,
                                        tile_flops * tiles));
  }
  return absl::OkStatus();
}

// MLX implicit_gemm_conv_2D_gpu, on input channels `c` (the padded count for
// kPadChannels).
absl::Status RunImplicit(rt::Device* device, rt::Stream* stream,
                         const ConvParams& p, const ConvPlan& plan, int64_t c,
                         const void* in, const void* weight, void* out) {
  const ConvTile& t = plan.tile;
  const SteelConvParams sp = MakeSteelParams(p, c);
  ImplicitGemmParams gp = {};
  gp.M = static_cast<int32_t>(OutPixels(p));
  gp.N = static_cast<int32_t>(p.o);
  gp.K = static_cast<int32_t>(p.kh * p.kw * c);
  const int64_t channel_k_iters = CeilDiv(c, t.bk);
  int64_t gemm_k_iters = p.kh * p.kw * channel_k_iters;
  if (c <= 2) {
    gemm_k_iters = CeilDiv(gp.K, t.bk);
  } else if (c <= 4) {
    gemm_k_iters = CeilDiv(p.kh * p.kw * 4, t.bk);
  }
  gp.gemm_k_iterations = static_cast<int32_t>(gemm_k_iters);
  const int64_t sign = p.flip ? -1 : 1;
  const int64_t ijw = sp.in_strides[2] * p.kdil[1];
  const int64_t ijh = sp.in_strides[1] * p.kdil[0];
  gp.inp_jump_w = static_cast<int32_t>(sign * ijw);
  gp.inp_jump_h = static_cast<int32_t>(sign * (ijh - (p.kw - 1) * ijw));
  gp.inp_jump_c = static_cast<int32_t>(t.bk - sign * (p.kh - 1) * ijh -
                                       sign * (p.kw - 1) * ijw);
  gp.tiles_n = static_cast<int32_t>(CeilDiv(p.o, t.bn));
  gp.tiles_m = static_cast<int32_t>(CeilDiv(gp.M, t.bm));
  gp.swizzle_log = 0;

  const ConvKernelSource k =
      ImplicitConvKernel(p.type, t, plan.n_channels, plan.small_filter);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* kernel,
                        device->GetKernel(k.msl, k.function, k.constants));
  return LaunchRowTiles(
      stream, *kernel, p, plan, gp, static_cast<uint32_t>(p.groups),
      {rt::KernelArg::Buffer(in), rt::KernelArg::Buffer(weight),
       rt::KernelArg::Buffer(out), rt::KernelArg::Bytes(&sp, sizeof(sp)),
       rt::KernelArg::Bytes(&gp, sizeof(gp))});
}

// MLX implicit_gemm_conv_2D_general_gpu.
absl::Status RunGeneral(rt::Device* device, rt::Stream* stream,
                        const ConvParams& p, const ConvPlan& plan,
                        const void* in, const void* weight, void* out) {
  const ConvTile& t = plan.tile;
  const SteelConvParams sp = MakeSteelParams(p, p.c);
  GeneralJumpParams jp = {};
  jp.f_wgt_jump_h = static_cast<int32_t>(
      std::lcm(p.idil[0], p.kdil[0]) / p.kdil[0]);
  jp.f_wgt_jump_w = static_cast<int32_t>(
      std::lcm(p.idil[1], p.kdil[1]) / p.kdil[1]);
  jp.f_out_jump_h = static_cast<int32_t>(
      std::lcm(p.idil[0], p.stride[0]) / p.stride[0]);
  jp.f_out_jump_w = static_cast<int32_t>(
      std::lcm(p.idil[1], p.stride[1]) / p.stride[1]);
  jp.adj_out_h = static_cast<int32_t>(CeilDiv(p.out_h, jp.f_out_jump_h));
  jp.adj_out_w = static_cast<int32_t>(CeilDiv(p.out_w, jp.f_out_jump_w));
  jp.adj_out_hw = jp.adj_out_h * jp.adj_out_w;
  jp.adj_implicit_m = static_cast<int32_t>(p.n * jp.adj_out_hw);

  // Per output phase: the first tap landing on a real input row / column
  // and how many taps (every f_wgt_jump-th) do.
  auto bases = [&](int64_t phases, int64_t stride, int64_t pad, int64_t k,
                   int64_t kdil, int64_t idil, int64_t f_wgt_jump) {
    std::vector<GeneralBaseInfo> out(phases);
    const int64_t jump = p.flip ? -kdil : kdil;
    const int64_t init = p.flip ? (k - 1) * kdil : 0;
    for (int64_t i = 0; i < phases; ++i) {
      int64_t pos = i * stride - pad + init;
      int64_t base = 0;
      // C++ % keeps the sign; the kernel's test is the same `!= 0`.
      while (base < k && pos % idil != 0) {
        ++base;
        pos += jump;
      }
      out[i] = {static_cast<int32_t>(base),
                static_cast<int32_t>(CeilDiv(k - base, f_wgt_jump))};
    }
    return out;
  };
  const std::vector<GeneralBaseInfo> base_h =
      bases(jp.f_out_jump_h, p.stride[0], p.pad_lo[0], p.kh, p.kdil[0],
            p.idil[0], jp.f_wgt_jump_h);
  const std::vector<GeneralBaseInfo> base_w =
      bases(jp.f_out_jump_w, p.stride[1], p.pad_lo[1], p.kw, p.kdil[1],
            p.idil[1], jp.f_wgt_jump_w);

  ImplicitGemmParams gp = {};
  gp.M = static_cast<int32_t>(OutPixels(p));
  gp.N = static_cast<int32_t>(p.o);
  gp.K = static_cast<int32_t>(p.kh * p.kw * p.c);
  gp.gemm_k_iterations = static_cast<int32_t>(CeilDiv(p.c, t.bk));
  gp.tiles_n = static_cast<int32_t>(CeilDiv(p.o, t.bn));
  gp.tiles_m = static_cast<int32_t>(CeilDiv(jp.adj_implicit_m, t.bm));
  gp.swizzle_log = 0;
  // inp_jump_* are unused by the general loaders.

  const ConvKernelSource k = GeneralConvKernel(p.type, t, plan.align_c);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* kernel,
                        device->GetKernel(k.msl, k.function, k.constants));
  return LaunchRowTiles(
      stream, *kernel, p, plan, gp,
      static_cast<uint32_t>(jp.f_out_jump_h * jp.f_out_jump_w),
      {rt::KernelArg::Buffer(in), rt::KernelArg::Buffer(weight),
       rt::KernelArg::Buffer(out), rt::KernelArg::Bytes(&sp, sizeof(sp)),
       rt::KernelArg::Bytes(&gp, sizeof(gp)),
       rt::KernelArg::Bytes(&jp, sizeof(jp)),
       rt::KernelArg::Bytes(base_h.data(),
                            base_h.size() * sizeof(GeneralBaseInfo)),
       rt::KernelArg::Bytes(base_w.data(),
                            base_w.size() * sizeof(GeneralBaseInfo))});
}

// dst[r, :cols_out] = [src[r, :cols], 0...] for r < rows.
absl::Status PadCols(rt::Device* device, rt::Stream* stream, ConvType type,
                     const void* src, void* dst, int64_t rows, int64_t cols,
                     int64_t out_cols) {
  const ConvKernelSource k = PadColsKernel(type);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* kernel,
                        device->GetKernel(k.msl, k.function, k.constants));
  const PadRows pr{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                   static_cast<uint32_t>(out_cols)};
  const rt::Dim3 threads{32, 8, 1};
  const rt::Dim3 groups{static_cast<uint32_t>(CeilDiv(out_cols, 32)),
                        static_cast<uint32_t>(CeilDiv(rows, 8)), 1};
  return rt::LaunchKernel(stream, *kernel, {src, dst}, pr, groups, threads);
}

absl::StatusOr<blas::MpsOperand> Operand(rt::Device* device, const void* ptr,
                                         ConvType type, int64_t ld,
                                         bool transpose) {
  ABSL_ASSIGN_OR_RETURN(rt::BufferRef ref, device->Resolve(ptr));
  blas::MpsOperand op;
  op.buffer = ref.buffer;
  op.offset = ref.offset;
  op.ld = ld;
  op.dtype = GemmType(type);
  op.transpose = transpose;
  return op;
}

// The widest vector the unfold copies in, 16, 8, 4 or 2 bytes: it divides
// a pixel's channels and both buffer offsets. Never below one element (the
// loop stops there), which divides both as the operands are arrays of it.
absl::StatusOr<int> UnfoldVectorBytes(rt::Device* device, const ConvParams& p,
                                      const void* in, const void* dst) {
  ABSL_ASSIGN_OR_RETURN(rt::BufferRef in_ref, device->Resolve(in));
  ABSL_ASSIGN_OR_RETURN(rt::BufferRef dst_ref, device->Resolve(dst));
  const int64_t item = ItemSize(p.type);
  const uint64_t offsets = in_ref.offset | dst_ref.offset;
  int64_t bytes = 16;
  while (bytes > item &&
         ((p.c * item) % bytes != 0 || offsets % bytes != 0)) {
    bytes /= 2;
  }
  return static_cast<int>(bytes);
}

// MLX explicit_gemm_conv_ND_gpu: per tile of unfold_rows output pixels,
// unfold them into the workspace ([rows, kH * kW * C]) and multiply by the
// weight ([O, kH * kW * C], transposed) into the output rows.
absl::Status RunExplicit(rt::Device* device, rt::Stream* stream,
                         const ConvParams& p, const ConvPlan& plan,
                         const void* in, const void* weight, void* out,
                         void* workspace) {
  const int64_t m = OutPixels(p);
  const int64_t k = p.kh * p.kw * p.c;
  ABSL_ASSIGN_OR_RETURN(blas::MpsOperand b,
                        Operand(device, weight, p.type, k, true));
  for (int64_t row = 0; row < m; row += plan.unfold_rows) {
    const int64_t rows = std::min(plan.unfold_rows, m - row);
    ABSL_RETURN_IF_ERROR(
        Unfold(device, stream, p, in, workspace, row, rows));
    blas::GemmParams g;
    g.m = rows;
    g.n = p.o;
    g.k = k;
    g.batch_count = 1;
    ABSL_ASSIGN_OR_RETURN(g.a, Operand(device, workspace, p.type, k, false));
    g.b = b;
    ABSL_ASSIGN_OR_RETURN(
        g.c, Operand(device,
                     static_cast<char*>(out) + row * p.o * ItemSize(p.type),
                     p.type, p.o, false));
    ABSL_RETURN_IF_ERROR(blas::RunSteelGemm(device, stream, g));
  }
  return absl::OkStatus();
}

// The weight gradient (see conv.h): per chunk of unfold_rows patch rows, one
// batched GEMM over its parts of split_rows rows (plus one for a shorter
// last part) accumulating dY^T x patches into the f32 partials, which
// sum_splits then adds into dW. With one part and one chunk the GEMM writes
// dW itself.
absl::Status RunWeightGrad(rt::Device* device, rt::Stream* stream,
                           const ConvParams& p, const ConvPlan& plan,
                           const void* in, const void* dy, void* dw,
                           void* workspace) {
  const int64_t m = OutPixels(p);
  const int64_t k = p.kh * p.kw * p.c;
  const int64_t item = ItemSize(p.type);
  if (m == 0) return stream->Memset8(dw, 0, p.o * k * item);
  const blas::SteelTile tile{plan.tile.bm, plan.tile.bn, plan.tile.bk,
                             plan.tile.wm, plan.tile.wn};
  char* patches = static_cast<char*>(workspace);
  const bool direct = plan.splits == 1 && plan.unfold_rows >= m;
  void* partials =
      direct ? dw : patches + WgradPartialsOffset(plan.unfold_rows, k, item);
  const blas::MpsDType partial_type =
      direct ? GemmType(p.type) : blas::MpsDType::kF32;
  for (int64_t row = 0; row < m; row += plan.unfold_rows) {
    const int64_t rows = std::min(plan.unfold_rows, m - row);
    ABSL_RETURN_IF_ERROR(Unfold(device, stream, p, in, patches, row, rows));
    // Parts [0, full) of split_rows rows, then the rest into part `full`,
    // which the first chunk (as long as any later one) also wrote.
    const int64_t full = rows / plan.split_rows;
    const int64_t rest = rows % plan.split_rows;
    for (int pass = 0; pass < 2; ++pass) {
      const int64_t batch = pass == 0 ? full : 1;
      const int64_t part = pass == 0 ? 0 : full;
      const int64_t len = pass == 0 ? plan.split_rows : rest;
      if (batch == 0 || len == 0) continue;
      blas::GemmParams g;
      g.m = p.o;
      g.n = k;
      g.k = len;
      g.batch_count = batch;
      g.beta = row == 0 ? 0.0 : 1.0;
      const char* dy_rows = static_cast<const char*>(dy) +
                            (row + part * plan.split_rows) * p.o * item;
      ABSL_ASSIGN_OR_RETURN(g.a, Operand(device, dy_rows, p.type, p.o, true));
      g.a.batch_stride = plan.split_rows * p.o;
      ABSL_ASSIGN_OR_RETURN(
          g.b, Operand(device, patches + part * plan.split_rows * k * item,
                       p.type, k, false));
      g.b.batch_stride = plan.split_rows * k;
      ABSL_ASSIGN_OR_RETURN(
          g.c, Operand(device,
                       static_cast<char*>(partials) +
                           part * p.o * k * (direct ? item : 4),
                       p.type, k, false));
      g.c.dtype = partial_type;
      g.c.batch_stride = p.o * k;
      ABSL_RETURN_IF_ERROR(blas::RunSteelGemm(device, stream, g, &tile));
    }
  }
  if (direct) return absl::OkStatus();
  const ConvKernelSource sk = SumSplitsKernel(p.type);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* sum,
                        device->GetKernel(sk.msl, sk.function, sk.constants));
  const SumSplits ss{static_cast<uint32_t>(p.o * k),
                     static_cast<uint32_t>(plan.splits)};
  return rt::LaunchKernel(stream, *sum, {partials, dw}, ss,
                          {static_cast<uint32_t>(CeilDiv(p.o * k, 256)), 1, 1},
                          {256, 1, 1});
}

}  // namespace

std::string ConvParamsDebugString(const ConvParams& p) {
  return absl::StrCat(
      MslTypeName(p.type), " in [", p.n, ",", p.h, ",", p.w, ",", p.c,
      "] weight [", p.o, ",", p.kh, ",", p.kw, ",", p.c / std::max<int64_t>(p.groups, 1),
      "] out [", p.n, ",", p.out_h, ",", p.out_w, ",", p.o, "] stride ",
      p.stride[0], "x", p.stride[1], " pad_lo ", p.pad_lo[0], "x",
      p.pad_lo[1], " kdil ", p.kdil[0], "x", p.kdil[1], " idil ", p.idil[0],
      "x", p.idil[1], " groups ", p.groups, p.flip ? " flip" : "");
}

const char* ConvPathName(ConvPath path) {
  switch (path) {
    case ConvPath::kImplicit:
      return "implicit";
    case ConvPath::kPadChannels:
      return "pad_channels";
    case ConvPath::kGeneral:
      return "general";
    case ConvPath::kExplicit:
      return "explicit";
    case ConvPath::kWeightGrad:
      return "weight_grad";
  }
  return "?";
}

uint64_t ConvFlops(const ConvParams& p) {
  return 2ull * static_cast<uint64_t>(OutPixels(p)) *
         static_cast<uint64_t>(p.o) * static_cast<uint64_t>(p.kh * p.kw) *
         static_cast<uint64_t>(p.c / std::max<int64_t>(p.groups, 1));
}

absl::StatusOr<ConvPlan> PlanConv(const ConvParams& p, const ConvPath* force,
                                  uint64_t max_launch_flops) {
  for (int64_t v : {p.n, p.h, p.w, p.c, p.o, p.kh, p.kw, p.out_h, p.out_w}) {
    if (v < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("negative size: ", ConvParamsDebugString(p)));
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (p.stride[i] < 1 || p.kdil[i] < 1 || p.idil[i] < 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "stride and dilations must be >= 1: ", ConvParamsDebugString(p)));
    }
  }
  if (p.groups != 1) return Unsupported(p, "grouped");
  if (p.pad_lo[0] < 0 || p.pad_lo[1] < 0) {
    return Unsupported(p, "negative low padding");
  }
  const bool wgrad = p.kind == ConvKind::kWeightGrad;
  // (The weight gradient's contraction is N * oH * oW; RunConv zero-fills
  // dW when that is empty.)
  if (!wgrad && (p.c == 0 || p.kh == 0 || p.kw == 0)) {
    return Unsupported(p, "empty contraction");
  }
  if (!wgrad && (p.h == 0 || p.w == 0)) return Unsupported(p, "empty input");
  // The kernels index within an operand in 32 bits (MLX convention); the
  // pad and dilation terms of the input positions too.
  const int64_t k = p.kh * p.kw * p.c;
  const int64_t m = OutPixels(p);
  for (int64_t v : {p.n * p.h * p.w * p.c, p.o * k, m * p.o, m, k,
                    ((p.h - 1) * p.idil[0] + 1) + p.pad_lo[0] +
                        (p.kh - 1) * p.kdil[0] + p.stride[0] * p.out_h,
                    ((p.w - 1) * p.idil[1] + 1) + p.pad_lo[1] +
                        (p.kw - 1) * p.kdil[1] + p.stride[1] * p.out_w}) {
    if (v > kInt32Max) return Unsupported(p, "exceeds 32-bit indexing");
  }
  const uint64_t max_flops = std::max<uint64_t>(max_launch_flops, 1);
  const int64_t item = ItemSize(p.type);

  if (wgrad) {
    if (force != nullptr && *force != ConvPath::kWeightGrad) {
      return Unsupported(p, absl::StrCat("path ", ConvPathName(*force)));
    }
    if (k > kSteelMaxLd || p.o > kSteelMaxLd) {
      return Unsupported(p, "a steel GEMM leading dimension over INT32_MAX/256");
    }
    ConvPlan plan;
    plan.path = ConvPath::kWeightGrad;
    if (m == 0 || p.o == 0 || k == 0) return plan;
    // The GEMM tile: 64 x 64 with 2 x 2 simdgroups (1.3-1.5x faster than
    // 32 x 32 on the CIFAR layers), 32 x 32 when O or K is small.
    plan.tile = p.o <= 32 || k <= 32 ? ConvTile{32, 32, 16, 2, 2}
                                     : ConvTile{64, 64, 16, 2, 2};
    // Chunks bounded in bytes and in flops (2 * O * K per row).
    const uint64_t row_bytes = static_cast<uint64_t>(k * item);
    const uint64_t row_flops = 2ull * static_cast<uint64_t>(p.o * k);
    const uint64_t max_rows =
        std::min(kUnfoldMaxBytes / row_bytes, max_flops / row_flops);
    plan.unfold_rows = std::clamp<int64_t>(max_rows, 1, m);
    // Parts: enough threadgroups over the GEMM's (O x K) tiles, each part
    // at least kMinSplitRows long, the partials within their cap.
    const uint64_t part_bytes = static_cast<uint64_t>(p.o * k) * 4;
    const int64_t tiles =
        CeilDiv(p.o, plan.tile.bm) * CeilDiv(k, plan.tile.bn);
    plan.splits = std::clamp<int64_t>(
        CeilDiv(kWgradTargetGroups, tiles), 1,
        std::max<int64_t>(plan.unfold_rows / kMinSplitRows, 1));
    plan.splits = std::clamp<int64_t>(kWgradMaxPartialsBytes / part_bytes, 1,
                                      plan.splits);
    // The patches and the partials together within kUnfoldMaxBytes when
    // the chunk alone reached it: the parts do not grow the workspace.
    if (plan.splits > 1) {
      const uint64_t partials = static_cast<uint64_t>(plan.splits) * part_bytes;
      plan.unfold_rows = std::clamp<int64_t>(
          std::min<uint64_t>(max_rows,
                             (kUnfoldMaxBytes - partials) / row_bytes),
          1, m);
    }
    plan.split_rows = CeilDiv(plan.unfold_rows, plan.splits);
    plan.splits = CeilDiv(plan.unfold_rows, plan.split_rows);
    const bool direct = plan.splits == 1 && plan.unfold_rows >= m;
    plan.workspace_bytes =
        direct ? plan.unfold_rows * row_bytes
               : WgradPartialsOffset(plan.unfold_rows, k, item) +
                     static_cast<uint64_t>(plan.splits) * part_bytes;
    if (plan.splits * p.o * k > kInt32Max) {
      return Unsupported(p, "exceeds 32-bit indexing");
    }
    return plan;
  }

  const bool idil1 = NoInputDilation(p);
  const int64_t padded_c = CeilDiv(p.c, 16) * 16;
  const bool implicit_ok =
      idil1 && ImplicitChannelsOk(p.c) && ImplicitOutputOk(p, p.c);
  const bool pad_ok = idil1 && Stride1(p) && !ImplicitChannelsOk(p.c) &&
                      ImplicitOutputOk(p, padded_c) &&
                      (p.n * p.h * p.w + p.o * p.kh * p.kw) * padded_c <=
                          kInt32Max;
  const bool out_large = m >= 256;

  ConvPath path;
  if (force != nullptr) {
    path = *force;
    const bool ok = path == ConvPath::kImplicit      ? implicit_ok
                    : path == ConvPath::kPadChannels ? pad_ok
                    : path == ConvPath::kWeightGrad  ? false
                                                     : true;
    if (!ok) {
      return Unsupported(p, absl::StrCat("path ", ConvPathName(path)));
    }
  } else if (pad_ok && out_large && p.kh * p.kw >= 9) {
    path = ConvPath::kPadChannels;
  } else if (implicit_ok) {
    path = ConvPath::kImplicit;
  } else if ((p.c % 16 == 0 && p.o % 16 == 0) || out_large) {
    path = ConvPath::kGeneral;
  } else {
    path = ConvPath::kExplicit;
  }

  ConvPlan plan;
  plan.path = path;
  int64_t tile_rows = m;  // GEMM rows the row tiles cover
  switch (path) {
    case ConvPath::kPadChannels:
    case ConvPath::kImplicit: {
      const int64_t c = path == ConvPath::kPadChannels ? padded_c : p.c;
      plan.tile = ImplicitTile(m, p.o, c);
      plan.n_channels = c <= 4 ? static_cast<int>(c) : 0;
      plan.small_filter = plan.n_channels == 0 && p.kh <= 16 && p.kw <= 16;
      if (path == ConvPath::kPadChannels) {
        plan.padded_c = padded_c;
        plan.workspace_bytes = static_cast<uint64_t>(
            (p.n * p.h * p.w + p.o * p.kh * p.kw) * padded_c * item);
      }
      break;
    }
    case ConvPath::kGeneral: {
      const int64_t phases_h =
          std::lcm(p.idil[0], p.stride[0]) / p.stride[0];
      const int64_t phases_w =
          std::lcm(p.idil[1], p.stride[1]) / p.stride[1];
      tile_rows =
          p.n * CeilDiv(p.out_h, phases_h) * CeilDiv(p.out_w, phases_w);
      plan.tile = GeneralTile(tile_rows, p.o, p.c);
      plan.align_c = p.c % plan.tile.bk == 0;
      break;
    }
    case ConvPath::kExplicit: {
      if (k > kSteelMaxLd || p.o > kSteelMaxLd) {
        return Unsupported(p,
                           "a steel GEMM leading dimension over INT32_MAX/256");
      }
      const uint64_t row_bytes = static_cast<uint64_t>(k * item);
      const uint64_t row_flops = 2ull * static_cast<uint64_t>(p.o * k);
      plan.unfold_rows = std::clamp<int64_t>(
          std::min(kUnfoldMaxBytes / row_bytes, max_flops / row_flops), 1,
          std::max<int64_t>(m, 1));
      plan.workspace_bytes = plan.unfold_rows * row_bytes;
      break;
    }
    case ConvPath::kWeightGrad:
      break;
  }
  if (path != ConvPath::kExplicit) {
    // The kernels' first row of a tile, tile * BM, is a 32-bit int.
    const int64_t tiles_m = CeilDiv(tile_rows, plan.tile.bm);
    if (tiles_m * plan.tile.bm > kInt32Max) {
      return Unsupported(p, "exceeds 32-bit indexing");
    }
    const uint64_t tile_flops =
        (PlannedFlops(p, plan) + tiles_m - 1) / std::max<int64_t>(tiles_m, 1);
    plan.tiles_m = tiles_m;
    plan.launch_tiles_m = std::clamp<int64_t>(
        max_flops / std::max<uint64_t>(tile_flops, 1), 1,
        std::max<int64_t>(tiles_m, 1));
  }
  return plan;
}

absl::Status Unfold(rt::Device* device, rt::Stream* stream,
                    const ConvParams& p, const void* in, void* dst,
                    int64_t row, int64_t rows) {
  // The kernel's thread index (one per vector, at most one per element) is
  // a 32-bit int; the paths' chunks are far below it.
  if (rows * p.kh * p.kw * p.c > kInt32Max) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unfold: ", rows, " rows exceed 32-bit indexing: ",
                     ConvParamsDebugString(p)));
  }
  ABSL_ASSIGN_OR_RETURN(const int bytes,
                        UnfoldVectorBytes(device, p, in, dst));
  const ConvKernelSource uk = UnfoldKernel(bytes);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* unfold,
                        device->GetKernel(uk.msl, uk.function, uk.constants));
  const SteelConvParams sp = MakeSteelParams(p, p.c);
  const UnfoldRows ur{static_cast<int32_t>(row), static_cast<int32_t>(rows),
                      static_cast<int32_t>(p.c * ItemSize(p.type) / bytes)};
  const int64_t vecs = rows * p.kh * p.kw * ur.vecs;
  return stream->Launch(
      *unfold, {static_cast<uint32_t>(CeilDiv(vecs, 256)), 1, 1},
      {256, 1, 1},
      {rt::KernelArg::Buffer(in), rt::KernelArg::Buffer(dst),
       rt::KernelArg::Bytes(&sp, sizeof(sp)),
       rt::KernelArg::Bytes(&ur, sizeof(ur))});
}

absl::Status RunConv(rt::Device* device, rt::Stream* stream,
                     const ConvParams& p, const ConvPlan& plan,
                     const void* in, const void* b, void* out,
                     void* workspace) {
  if (plan.workspace_bytes > 0 && workspace == nullptr) {
    return absl::InvalidArgumentError("RunConv: missing workspace");
  }
  if ((plan.path == ConvPath::kWeightGrad) !=
      (p.kind == ConvKind::kWeightGrad)) {
    return absl::InvalidArgumentError("RunConv: plan of another kind");
  }
  if (p.kind == ConvKind::kWeightGrad) {
    if (p.o == 0 || p.kh * p.kw * p.c == 0) return absl::OkStatus();
    return RunWeightGrad(device, stream, p, plan, in, b, out, workspace);
  }
  const void* weight = b;
  if (OutPixels(p) == 0 || p.o == 0) return absl::OkStatus();
  switch (plan.path) {
    case ConvPath::kImplicit:
      return RunImplicit(device, stream, p, plan, p.c, in, weight, out);
    case ConvPath::kPadChannels: {
      const int64_t cp = plan.padded_c;
      char* in_p = static_cast<char*>(workspace);
      char* wt_p = in_p + p.n * p.h * p.w * cp * ItemSize(p.type);
      ABSL_RETURN_IF_ERROR(PadCols(device, stream, p.type, in, in_p,
                                   p.n * p.h * p.w, p.c, cp));
      ABSL_RETURN_IF_ERROR(PadCols(device, stream, p.type, weight, wt_p,
                                   p.o * p.kh * p.kw, p.c, cp));
      return RunImplicit(device, stream, p, plan, cp, in_p, wt_p, out);
    }
    case ConvPath::kGeneral:
      return RunGeneral(device, stream, p, plan, in, weight, out);
    case ConvPath::kExplicit:
      return RunExplicit(device, stream, p, plan, in, weight, out, workspace);
    case ConvPath::kWeightGrad:
      break;
  }
  return absl::InternalError("RunConv: unknown path");
}

}  // namespace conv
}  // namespace metal_pjrt
