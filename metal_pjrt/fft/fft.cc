// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/fft.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.

#include "metal_pjrt/fft/fft.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/kernels/fft.metal.h"
#include "metal_pjrt/kernels/fft_misc.metal.h"
#include "metal_pjrt/runtime/kernel_launch.h"

namespace metal_pjrt {
namespace fft {
namespace {

// Must match FftMiscParams in kernels/fft_misc.metal.
struct FftMiscParams {
  uint32_t rows, n, in_len, out_len, pad_len;
};

constexpr uint32_t kMiscThreads = 256;

const char* IoType(bool real) { return real ? "float" : "float2"; }

bool IsPowerOf2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// MLX's kernel names (fft.cpp 695-717, without the per-size suffix: the
// function constants tell those apart).
std::string KernelName(FftFunction f, int tg_mem_size, bool real_in,
                       bool real_out, int step, bool four_step_real) {
  const char* prefix = f == FftFunction::kStockham  ? "fft_mem_"
                       : f == FftFunction::kRader   ? "rader_fft_mem_"
                       : f == FftFunction::kBluestein ? "bluestein_fft_mem_"
                                                      : "four_step_mem_";
  std::string name = absl::StrCat(prefix, tg_mem_size, "_", IoType(real_in),
                                  "_", IoType(real_out));
  if (f == FftFunction::kFourStep) {
    absl::StrAppend(&name, "_", step, "_", four_step_real ? "true" : "false");
  }
  return name;
}

FftFunction SingleKernelFunction(const FftPlan& plan) {
  return plan.bluestein_n > 0 ? FftFunction::kBluestein
         : plan.rader_n > 1   ? FftFunction::kRader
                              : FftFunction::kStockham;
}

rt::KernelArg Int(int64_t v) {
  const int32_t i = static_cast<int32_t>(v);
  return rt::KernelArg::Bytes(&i, sizeof(i));
}

// Whether `c` has the tables `plan` reads, in MakeFftConstants(plan)'s
// order, one after the other within the bytes; no bytes for a plan without
// tables.
bool ConstantsMatch(const FftPlan& plan, const FftConstants& c) {
  // Tables of these sizes at these offsets, in order, without overlap.
  auto laid_out = [&](std::initializer_list<std::pair<uint64_t, uint64_t>>
                          tables) {
    uint64_t end = 0;
    for (auto [offset, bytes] : tables) {
      if (offset < end || bytes > c.size || offset > c.size - bytes) {
        return false;
      }
      end = offset + bytes;
    }
    return true;
  };
  if (plan.bluestein_n > 0) {
    return laid_out({{c.w_q, uint64_t(plan.bluestein_n) * 8},
                     {c.w_k, uint64_t(plan.n) * 8}});
  }
  if (plan.rader_n > 1) {
    const uint64_t m = plan.rader_n - 1;
    return laid_out({{c.b_q, m * 8}, {c.g_q, m * 2}, {c.g_minus_q, m * 2}});
  }
  return c.size == 0;
}

const void* Offset(const void* p, uint64_t bytes) {
  return static_cast<const char*>(p) + bytes;
}
void* Offset(void* p, uint64_t bytes) { return static_cast<char*>(p) + bytes; }

// One FFT kernel launch (fft.cpp 621-813): `plan` is the plan of the
// launch's length, `sequences` the transforms of that length, `args` the
// kernel's arguments in order.
absl::Status LaunchFftKernel(rt::Device* device, rt::Stream* stream,
                             FftFunction f, const FftPlan& plan, bool inverse,
                             bool real_in, bool real_out, int step,
                             bool four_step_real, int64_t sequences,
                             const std::vector<rt::KernelArg>& args,
                             uint64_t flops) {
  const bool four_step = f == FftFunction::kFourStep;
  const FftLaunchGeometry g =
      LaunchGeometry(plan, real_in || real_out, four_step, sequences);
  const FftKernelSource src = FftKernel(f, g.tg_mem_size, real_in, real_out,
                                        step, four_step_real, plan, inverse);
  ABSL_ASSIGN_OR_RETURN(
      const rt::Kernel* kernel,
      device->GetKernel(src.msl, src.function, src.constants));
  // MLX dispatches threads (groups, batch_per_group, threads_per_fft) in
  // threadgroups of (1, batch_per_group, threads_per_fft).
  return stream->Launch(
      *kernel, {static_cast<uint32_t>(g.groups), 1, 1},
      {1, static_cast<uint32_t>(g.batch_per_group),
       static_cast<uint32_t>(g.threads_per_fft)},
      args, /*threadgroup_memory_bytes=*/0, flops);
}

absl::Status LaunchMisc(rt::Device* device, rt::Stream* stream,
                        const std::string& function,
                        const std::vector<const void*>& buffers,
                        const FftMiscParams& p, uint32_t width,
                        uint32_t height) {
  const FftKernelSource src = FftMiscKernel(function);
  ABSL_ASSIGN_OR_RETURN(const rt::Kernel* kernel,
                        device->GetKernel(src.msl, src.function));
  return rt::LaunchKernel(stream, *kernel, buffers, p,
                          {(width + kMiscThreads - 1) / kMiscThreads, height,
                           1},
                          {kMiscThreads, 1, 1});
}

// four_step_fft (fft.cpp 497-525) for a power of two: pass 0 (n1-point
// transforms, strided, times the twiddles) from `in` to `temp`, pass 1
// (n2-point transforms) from `temp` to `out`. `in` may be `out`.
absl::Status FourStep(rt::Device* device, rt::Stream* stream,
                      const FftPlan& plan, bool inverse, bool rfft, bool irfft,
                      int64_t rows, const void* in, void* temp, void* out) {
  for (int step : {0, 1}) {
    const int len = step == 0 ? plan.n1 : plan.n2;
    ABSL_ASSIGN_OR_RETURN(const FftPlan step_plan, PlanFft(len));
    const int64_t sequences = rows * plan.n / len;
    const std::vector<rt::KernelArg> args = {
        rt::KernelArg::Buffer(step == 0 ? in : temp),
        rt::KernelArg::Buffer(step == 0 ? temp : out), Int(plan.n1),
        Int(plan.n2), Int(sequences)};
    ABSL_RETURN_IF_ERROR(LaunchFftKernel(
        device, stream, FftFunction::kFourStep, step_plan, inverse,
        /*real_in=*/step == 0 && rfft, /*real_out=*/step == 1 && irfft, step,
        /*four_step_real=*/rfft || irfft, sequences, args,
        FftFlops(len, sequences)));
  }
  return absl::OkStatus();
}

// `rows` rows of the transform (one chunk).
absl::Status RunChunk(rt::Device* device, rt::Stream* stream,
                      const FftPlan& plan, FftType type, int64_t rows,
                      const FftConstants& constants,
                      const void* constants_device, const void* in, void* out,
                      void* workspace) {
  const int64_t n = plan.n;
  const bool inverse = IsInverse(type);
  const bool real_in = type == FftType::kRfft;
  const bool real_out = type == FftType::kIrfft;
  FftMiscParams mp;
  mp.rows = static_cast<uint32_t>(rows);
  mp.n = static_cast<uint32_t>(n);
  mp.in_len = static_cast<uint32_t>(InputLength(type, n));
  mp.out_len = static_cast<uint32_t>(OutputLength(type, n));
  mp.pad_len = static_cast<uint32_t>(std::max(plan.bluestein_n, 0));

  if (n == 1) {
    // A length-1 transform is the identity (MLX: copy_shared_buffer).
    if (real_in || real_out) {
      return LaunchMisc(device, stream,
                        real_in ? "fft_real_to_complex" : "fft_complex_to_real",
                        {in, out}, mp, mp.rows, 1);
    }
    return stream->MemcpyDeviceToDevice(out, in, rows * 8);
  }

  if (!plan.four_step) {
    const FftFunction f = SingleKernelFunction(plan);
    std::vector<rt::KernelArg> args = {rt::KernelArg::Buffer(in),
                                       rt::KernelArg::Buffer(out)};
    uint64_t flops = 0;
    if (f == FftFunction::kBluestein) {
      args.push_back(
          rt::KernelArg::Buffer(Offset(constants_device, constants.w_q)));
      args.push_back(
          rt::KernelArg::Buffer(Offset(constants_device, constants.w_k)));
      args.push_back(Int(n));
      args.push_back(Int(plan.bluestein_n));
      args.push_back(Int(rows));
      flops = 2 * FftFlops(plan.bluestein_n, rows);
    } else if (f == FftFunction::kRader) {
      args.push_back(
          rt::KernelArg::Buffer(Offset(constants_device, constants.b_q)));
      args.push_back(
          rt::KernelArg::Buffer(Offset(constants_device, constants.g_q)));
      args.push_back(
          rt::KernelArg::Buffer(Offset(constants_device, constants.g_minus_q)));
      args.push_back(Int(n));
      args.push_back(Int(rows));
      args.push_back(Int(plan.rader_n));
      flops = 3 * FftFlops(n, rows);
    } else {
      args.push_back(Int(n));
      args.push_back(Int(rows));
      flops = FftFlops(n, rows);
    }
    return LaunchFftKernel(device, stream, f, plan, inverse, real_in,
                           real_out, /*step=*/0, /*four_step_real=*/false,
                           rows, args, flops);
  }

  if (plan.bluestein_n < 0) {
    return FourStep(device, stream, plan, inverse, real_in, real_out, rows,
                    in, workspace, out);
  }

  // multi_upload_bluestein_fft (fft.cpp 372-495): out = w_k *
  // ifft(fft(pad(w_k * z)) * w_q)[n - 1:2n - 1], with z the input row as
  // complex (conjugated when inverse) and the output conjugated / its real
  // part taken and scaled by 1 / n when inverse; bluestein_pre and
  // bluestein_post fuse MLX's copies, multiplies, pad and slices.
  const int64_t bn = plan.bluestein_n;
  void* a = workspace;
  void* c = Offset(workspace, rows * bn * 8);
  const void* w_q = Offset(constants_device, constants.w_q);
  const void* w_k = Offset(constants_device, constants.w_k);
  const char* name = FftTypeName(type);
  ABSL_ASSIGN_OR_RETURN(const FftPlan pad_plan, PlanFft(bn));
  ABSL_RETURN_IF_ERROR(LaunchMisc(device, stream,
                                  absl::StrCat("bluestein_pre_", name),
                                  {in, w_k, a}, mp, mp.pad_len, mp.rows));
  ABSL_RETURN_IF_ERROR(FourStep(device, stream, pad_plan, /*inverse=*/false,
                                false, false, rows, a, c, a));
  ABSL_RETURN_IF_ERROR(LaunchMisc(device, stream, "bluestein_mul", {w_q, a},
                                  mp, mp.pad_len, mp.rows));
  ABSL_RETURN_IF_ERROR(FourStep(device, stream, pad_plan, /*inverse=*/true,
                                false, false, rows, a, c, a));
  return LaunchMisc(device, stream, absl::StrCat("bluestein_post_", name),
                    {a, w_k, out}, mp, mp.out_len, mp.rows);
}

}  // namespace

uint64_t FftFlops(int64_t length, int64_t transforms) {
  int log2 = 0;
  while ((int64_t{1} << log2) < length) ++log2;
  return 5ull * length * log2 * transforms;
}

FftKernelSource FftKernel(FftFunction f, int tg_mem_size, bool real_in,
                          bool real_out, int step, bool four_step_real,
                          const FftPlan& plan, bool inverse) {
  const char* func = f == FftFunction::kStockham    ? "fft"
                     : f == FftFunction::kRader     ? "rader_fft"
                     : f == FftFunction::kBluestein ? "bluestein_fft"
                                                    : "four_step_fft";
  const std::string name = KernelName(f, tg_mem_size, real_in, real_out, step,
                                      four_step_real);
  std::string inst =
      absl::StrCat("\ninstantiate_fft_kernel(\"", name, "\", ", func, ", ",
                   tg_mem_size, ", ", IoType(real_in), ", ", IoType(real_out));
  if (f == FftFunction::kFourStep) {
    absl::StrAppend(&inst, ", ", step, ", ", four_step_real ? "true" : "false");
  }
  absl::StrAppend(&inst, ")\n");

  // fft.cpp 624-667: 0 inverse, 1 power of two, 2 elements per thread,
  // 3 rader_m, 4-12 Stockham steps, 13-21 Rader steps, 22 the Bluestein
  // twiddle table (never used here).
  const int fft_size = plan.bluestein_n > 0 ? plan.bluestein_n : plan.n;
  std::vector<rt::FunctionConstant> constants = {
      rt::FunctionConstant::Bool(0, inverse),
      rt::FunctionConstant::Bool(1, IsPowerOf2(fft_size))};
  for (int i = 0; i < kNumRadices; ++i) {
    constants.push_back(rt::FunctionConstant::Int(4 + i, plan.stockham[i]));
  }
  for (int i = 0; i < kNumRadices; ++i) {
    constants.push_back(rt::FunctionConstant::Int(13 + i, plan.rader[i]));
  }
  constants.push_back(rt::FunctionConstant::Int(2, ElemsPerThread(plan)));
  constants.push_back(rt::FunctionConstant::Int(3, plan.n / plan.rader_n));
  if (plan.bluestein_n > 0 && !real_in && !real_out) {
    constants.push_back(rt::FunctionConstant::Bool(22, false));
  }
  return {absl::StrCat(kernels::kFftMsl, inst), name, std::move(constants)};
}

FftKernelSource FftMiscKernel(const std::string& function) {
  return {kernels::kFftMiscMsl, function, {}};
}

std::vector<FftKernelSource> AllFftKernels() {
  std::vector<FftKernelSource> out;
  std::set<std::string> seen;
  auto add = [&](FftFunction f, const FftPlan& plan, bool real_in,
                 bool real_out, int step, bool four_step_real) {
    const bool four_step = f == FftFunction::kFourStep;
    const FftLaunchGeometry g =
        LaunchGeometry(plan, real_in || real_out, four_step, 1);
    const std::string name = KernelName(f, g.tg_mem_size, real_in, real_out,
                                        step, four_step_real);
    if (!seen.insert(name).second) return;
    out.push_back(FftKernel(f, g.tg_mem_size, real_in, real_out, step,
                            four_step_real, plan, /*inverse=*/false));
  };
  // The single-kernel plans.
  for (int n = 2; n <= kMaxStockhamSize; ++n) {
    const FftPlan plan = *PlanFft(n);
    if (plan.four_step) continue;
    for (auto [real_in, real_out] :
         {std::pair{false, false}, {true, false}, {false, true}}) {
      add(SingleKernelFunction(plan), plan, real_in, real_out, 0, false);
    }
  }
  // The four-step passes (also those of the multi-upload Bluestein, c2c).
  for (int64_t n = 2 * kMaxStockhamSize; n <= kMaxFftSize; n *= 2) {
    const FftPlan plan = *PlanFft(n);
    const FftPlan p0 = *PlanFft(plan.n1), p1 = *PlanFft(plan.n2);
    add(FftFunction::kFourStep, p0, false, false, 0, false);
    add(FftFunction::kFourStep, p1, false, false, 1, false);
    add(FftFunction::kFourStep, p0, true, false, 0, true);
    add(FftFunction::kFourStep, p1, false, false, 1, true);
    add(FftFunction::kFourStep, p0, false, false, 0, true);
    add(FftFunction::kFourStep, p1, false, true, 1, true);
  }
  for (const char* name :
       {"fft_real_to_complex", "fft_complex_to_real", "bluestein_mul",
        "bluestein_pre_fft", "bluestein_pre_ifft", "bluestein_pre_rfft",
        "bluestein_pre_irfft", "bluestein_post_fft", "bluestein_post_ifft",
        "bluestein_post_rfft", "bluestein_post_irfft"}) {
    out.push_back(FftMiscKernel(name));
  }
  return out;
}

absl::Status RunFft(rt::Device* device, rt::Stream* stream,
                    const FftPlan& plan, FftType type, int64_t rows,
                    const FftConstants& constants,
                    const void* constants_device, const void* in, void* out,
                    void* workspace, uint64_t workspace_bytes,
                    int64_t max_chunk_elements) {
  if (rows < 0 || max_chunk_elements < 1) {
    return absl::InvalidArgumentError("FFT: negative rows or chunk size");
  }
  if (rows == 0) return absl::OkStatus();
  const uint64_t need = FftWorkspaceBytes(plan, rows, max_chunk_elements);
  if (workspace_bytes < need || (need > 0 && workspace == nullptr)) {
    return absl::InvalidArgumentError(
        absl::StrCat("FFT of length ", plan.n, " x ", rows, " rows needs ",
                     need, " bytes of workspace, got ", workspace_bytes));
  }
  // Constants of another plan (or none for a Rader or Bluestein plan) would
  // make the kernels read outside them, or from a null base: a GPU fault.
  if (!ConstantsMatch(plan, constants)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "FFT: ", constants.size,
        " bytes of constants do not match the plan of length ", plan.n));
  }
  if (constants.size > 0 && constants_device == nullptr) {
    return absl::InvalidArgumentError("FFT: constants not on the device");
  }
  const int64_t in_row = InputLength(type, plan.n) *
                         (type == FftType::kRfft ? 4 : 8);
  const int64_t out_row = OutputLength(type, plan.n) *
                          (type == FftType::kIrfft ? 4 : 8);
  const int64_t chunk = ChunkRows(plan, rows, max_chunk_elements);
  for (int64_t r = 0; r < rows; r += chunk) {
    ABSL_RETURN_IF_ERROR(RunChunk(device, stream, plan, type,
                                  std::min(chunk, rows - r), constants,
                                  constants_device, Offset(in, r * in_row),
                                  Offset(out, r * out_row), workspace));
  }
  return absl::OkStatus();
}

}  // namespace fft
}  // namespace metal_pjrt
