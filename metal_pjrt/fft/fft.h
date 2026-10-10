// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/fft.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.
//
// 1-D FFTs on the GPU with MLX's kernels (kernels/fft.metal, plan in
// fft_plan.h), no XLA, so fft_test runs them against rt::Device alone.
//
// RunFft transforms `rows` contiguous rows (row-major, the transform over
// the last dimension): complex64 is float2, the input rows have
// InputLength(type, n) elements and the output rows OutputLength(type, n).
// It follows MLX's fft_op (mlx/backend/metal/fft.cpp 531-816) for a
// contiguous last axis: one kernel for the Stockham, Rader and fused
// Bluestein plans; two strided passes through the workspace for the
// four-step plan; for the multi-upload Bluestein plan, elementwise kernels
// (kernels/fft_misc.metal) around a forward and an inverse four-step FFT of
// length bluestein_n, both in the workspace. Length 1 is a copy.
//
// The rows go in chunks of ChunkRows (fft_plan.h), each chunk through all of
// its passes before the next, so every launch is bounded (and so is the
// workspace). Every launch of an FFT kernel is charged 5 L log2 L flops per
// transform of length L it computes (rt::Stream::Launch), like a GEMM.
#ifndef METAL_PJRT_FFT_FFT_H_
#define METAL_PJRT_FFT_FFT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "metal_pjrt/fft/fft_plan.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace fft {

// The kernel templates of kernels/fft.metal.
enum class FftFunction { kStockham, kRader, kBluestein, kFourStep };

// One compilable variant: kernels/fft.metal plus one instantiation, the
// function name (MLX's kernel names) and the function constants.
struct FftKernelSource {
  std::string msl;
  std::string function;
  std::vector<rt::FunctionConstant> constants;
};

// The variant for a launch of `f` with `tg_mem_size` complex64s of
// threadgroup memory, real (float) or complex (float2) input and output,
// and for kFourStep the pass (`step` 0 or 1) and whether it is part of a
// real transform; specialized (function constants, fft.cpp 624-667) for
// `plan`, the plan of the launch's length, and the direction.
FftKernelSource FftKernel(FftFunction f, int tg_mem_size, bool real_in,
                          bool real_out, int step, bool four_step_real,
                          const FftPlan& plan, bool inverse);

// A kernel of kernels/fft_misc.metal by name.
FftKernelSource FftMiscKernel(const std::string& function);

// Every variant the dispatch can select, each once (with the constants of
// the first length that selects it), and every fft_misc kernel:
// kernels_test compiles them all.
std::vector<FftKernelSource> AllFftKernels();

// Enqueues the transform on the stream. `plan` is PlanFft(n); `constants`
// is MakeFftConstants(plan) (InvalidArgument if they do not match; the host
// bytes are not read) and `constants_device` a device copy of its bytes
// (unused when empty); `workspace` holds at least
// FftWorkspaceBytes(plan, rows, max_chunk_elements) bytes (InvalidArgument
// otherwise; unused when 0). Nothing is launched for rows == 0.
// `max_chunk_elements`: tests pass small values to force several chunks.
absl::Status RunFft(rt::Device* device, rt::Stream* stream,
                    const FftPlan& plan, FftType type, int64_t rows,
                    const FftConstants& constants,
                    const void* constants_device, const void* in, void* out,
                    void* workspace, uint64_t workspace_bytes,
                    int64_t max_chunk_elements = kMaxChunkElements);

// 5 L log2 L per transform of length L (the usual FFT flop count).
uint64_t FftFlops(int64_t length, int64_t transforms);

}  // namespace fft
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFT_FFT_H_
