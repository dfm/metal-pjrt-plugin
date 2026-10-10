// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/matmul.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.

#include "metal_pjrt/blas/gemv.h"

#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/kernels/gemv.metal.h"

namespace metal_pjrt {
namespace blas {
namespace {

// Must match GemvParams in kernels/gemv.metal.
struct GemvParams {
  int32_t K, rows, vecs;
  int32_t mat_ld, vec_ld;
  int32_t out_vec_stride, out_row_stride;
  int32_t bias_by_row;
  int64_t batch_stride_mat, batch_stride_vec, batch_stride_d;
  float alpha, beta;
};
static_assert(sizeof(GemvParams) == 64, "layout must match the MSL");

const char* MslType(MpsDType t) {
  switch (t) {
    case MpsDType::kF32:
      return "float";
    case MpsDType::kF16:
      return "half";
    case MpsDType::kBF16:
      return "bfloat";
  }
  return "?";
}

constexpr int64_t kMaxInt = std::numeric_limits<int32_t>::max();

// The matrix and vector operands of `p` under `plan`.
const MpsOperand& Matrix(const GemmParams& p, const GemvPlan& plan) {
  return plan.vectors_are_b ? p.a : p.b;
}
const MpsOperand& Vectors(const GemmParams& p, const GemvPlan& plan) {
  return plan.vectors_are_b ? p.b : p.a;
}

const void* DevicePtr(const MpsOperand& x) {
  auto* buf = static_cast<MTL::Buffer*>(x.buffer);
  return static_cast<const char*>(buf->contents()) + x.offset;
}

}  // namespace

std::optional<GemvPlan> ChooseGemv(const GemmParams& p, int gpu_family) {
  const MpsDType t = p.a.dtype;
  if (gpu_family < 9 || (t != MpsDType::kF16 && t != MpsDType::kBF16) ||
      p.b.dtype != t || (p.c.dtype != t && p.c.dtype != MpsDType::kF32) ||
      p.c.transpose || p.a.transpose || !p.b.transpose) {
    return std::nullopt;
  }
  GemvPlan plan;
  plan.vectors_are_b = p.n < p.m;
  const int64_t vecs = std::min(p.m, p.n);
  const int64_t rows = std::max(p.m, p.n);
  if (vecs < 2 || vecs > kGemvMaxVectors || p.k < kGemvMinK ||
      p.k % 4 != 0 || rows > kMaxInt || p.k > kMaxInt || p.batch_count < 1 ||
      p.batch_count > kMaxInt) {
    return std::nullopt;
  }
  // vec<T, 4> loads: 8-byte aligned rows (K-contiguous operands).
  const bool batched = p.batch_count > 1;
  for (const MpsOperand* x : {&p.a, &p.b}) {
    if (x->ld % 4 != 0 || x->ld > kMaxInt || x->offset % 8 != 0 ||
        (batched && x->batch_stride % 4 != 0) || x->batch_stride < 0) {
      return std::nullopt;
    }
  }
  const MpsOperand& v = Vectors(p, plan);
  if (v.ld < p.k || Matrix(p, plan).ld < p.k || p.c.ld < p.n ||
      p.c.ld > kMaxInt || p.c.batch_stride < 0 ||
      (vecs - 1) * v.ld + p.k > kMaxInt) {
    return std::nullopt;
  }
  // MLX: a register tile holds at most five vectors; M balances across
  // passes that each re-stream the matrix (at most 2 here).
  const int passes = static_cast<int>((vecs + 4) / 5);
  plan.vecs_per_tg = static_cast<int>((vecs + passes - 1) / passes);
  // Full rows double the threads per group and halve each lane's K share,
  // where the grid is thinnest; a vocabulary-wide matrix saturates the GPU
  // with one grid column.
  plan.k_lanes = passes == 1 || rows <= 64 ? 32 : 16;
  plan.grid_x = rows >= 65536 ? 1 : passes;
  return plan;
}

GemvKernelSource GemvKernel(MpsDType in, MpsDType out, int vecs_per_tg,
                            int k_lanes) {
  const std::string name =
      absl::StrCat("gemv_wide_", MslType(in), "_", MslType(out), "_nv",
                   vecs_per_tg, "_kl", k_lanes);
  return {absl::StrCat(kernels::kGemvMsl, "\ninstantiate_gemv_wide(\"", name,
                       "\", ", MslType(in), ", ", MslType(out), ", ",
                       vecs_per_tg, ", ", k_lanes, ")\n"),
          name};
}

std::vector<rt::FunctionConstant> GemvConstants(bool use_c,
                                                const SteelEpilogue& epi) {
  // The function_constant indices of kernels/gemv.metal.
  using C = rt::FunctionConstant;
  return {C::Bool(0, use_c), C::Bool(1, epi.bias != nullptr),
          C::Bool(2, epi.aux != nullptr), C::Int(3, epi.act)};
}

absl::Status RunGemv(rt::Device* device, rt::Stream* stream,
                     const GemmParams& p, const GemvPlan& plan,
                     const SteelEpilogue* epi) {
  const SteelEpilogue no_epi;
  if (epi == nullptr) epi = &no_epi;
  if (epi->act < 0 || epi->act > 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("RunGemv: unknown activation ", epi->act));
  }
  // The plan must be the one ChooseGemv makes for p (on a capable GPU).
  const std::optional<GemvPlan> expect = ChooseGemv(p, /*gpu_family=*/9);
  if (!expect.has_value() || expect->vectors_are_b != plan.vectors_are_b ||
      expect->vecs_per_tg != plan.vecs_per_tg ||
      expect->k_lanes != plan.k_lanes || expect->grid_x != plan.grid_x) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RunGemv: the plan does not apply: ", GemmParamsDebugString(p)));
  }
  for (const MpsOperand* x : {&p.a, &p.b, &p.c}) {
    if (x->buffer == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("RunGemv: null buffer: ", GemmParamsDebugString(p)));
    }
  }

  const GemvKernelSource source =
      GemvKernel(p.a.dtype, p.c.dtype, plan.vecs_per_tg, plan.k_lanes);
  ABSL_ASSIGN_OR_RETURN(
      const rt::Kernel* kernel,
      device->GetKernel(source.msl, source.function,
                        GemvConstants(p.beta != 0.0, *epi)));

  const MpsOperand& mat = Matrix(p, plan);
  const MpsOperand& vec = Vectors(p, plan);
  const bool batched = p.batch_count > 1;
  GemvParams gp;
  gp.K = static_cast<int32_t>(p.k);
  gp.rows = static_cast<int32_t>(plan.vectors_are_b ? p.m : p.n);
  gp.vecs = static_cast<int32_t>(plan.vectors_are_b ? p.n : p.m);
  gp.mat_ld = static_cast<int32_t>(mat.ld);
  gp.vec_ld = static_cast<int32_t>(vec.ld);
  // D[i, j] is at i * ldc + j; the vectors index i (m small) or j (n small).
  gp.out_vec_stride = plan.vectors_are_b ? 1 : static_cast<int32_t>(p.c.ld);
  gp.out_row_stride = plan.vectors_are_b ? static_cast<int32_t>(p.c.ld) : 1;
  gp.bias_by_row = plan.vectors_are_b ? 0 : 1;  // bias follows D's column
  gp.batch_stride_mat = batched ? mat.batch_stride : 0;
  gp.batch_stride_vec = batched ? vec.batch_stride : 0;
  gp.batch_stride_d = batched ? p.c.batch_stride : 0;
  gp.alpha = static_cast<float>(p.alpha);
  gp.beta = static_cast<float>(p.beta);

  constexpr int kRowsPerThreadgroup = 4;  // (32 / KL) * (KL / 8)
  rt::Dim3 groups{static_cast<uint32_t>(plan.grid_x),
                  static_cast<uint32_t>((gp.rows + kRowsPerThreadgroup - 1) /
                                        kRowsPerThreadgroup),
                  static_cast<uint32_t>(p.batch_count)};
  rt::Dim3 threads{static_cast<uint32_t>(4 * plan.k_lanes), 1, 1};
  const void* d = DevicePtr(p.c);
  // Unused epilogue buffers bind D (never accessed).
  const void* bias = epi->bias != nullptr ? epi->bias : d;
  const void* aux = epi->aux != nullptr ? epi->aux : d;
  return stream->Launch(*kernel, groups, threads,
                        {rt::KernelArg::Buffer(DevicePtr(mat)),
                         rt::KernelArg::Buffer(DevicePtr(vec)),
                         rt::KernelArg::Buffer(d), rt::KernelArg::Buffer(bias),
                         rt::KernelArg::Buffer(aux),
                         rt::KernelArg::Bytes(&gp, sizeof(gp))},
                        /*threadgroup_memory_bytes=*/0, GemmWork(p));
}

}  // namespace blas
}  // namespace metal_pjrt
