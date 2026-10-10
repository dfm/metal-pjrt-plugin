// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Parts of this file are ported from MLX (mlx/backend/metal/fft.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.
//
// The host side of the FFTs (fft.h), no Metal and no XLA: MLX's plan
// (mlx/backend/metal/fft.cpp, MLX 0.32.2; line numbers below refer to that
// file), the launch geometry, the Rader and Bluestein constants (computed in
// double) and the workspace size.
//
// A plan is one of MLX's paths for a length n:
//   Stockham        n = 2^a 3^b 5^c 7^d 11^e 13^f <= 4096: one kernel, the
//                   whole transform in threadgroup memory;
//   Rader           n <= 2048 with one prime factor p > 13 whose p - 1 is
//                   13-smooth: one kernel (rader_n = p);
//   fused Bluestein n <= 2048 otherwise: one kernel, a convolution of length
//                   bluestein_n = next_pow2(2n - 1) <= 4096;
//   four-step       a power of two 4096 < n <= 2^24: two strided passes of
//                   lengths n1 and n2 through the workspace;
//   multi-upload    any other 2048 < n: Bluestein over four-step passes of
//   Bluestein       length bluestein_n = next_pow2(2n - 1) <= 2^24, with
//                   elementwise kernels (kernels/fft_misc.metal) around them.
#ifndef METAL_PJRT_FFT_FFT_PLAN_H_
#define METAL_PJRT_FFT_FFT_PLAN_H_

#include <array>
#include <complex>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"

namespace metal_pjrt {
namespace fft {

// XLA's FftType. n is the transform length (fft_length): kFft/kIfft map
// complex64[n] rows to complex64[n], kRfft float32[n] to complex64[n/2+1],
// kIrfft complex64[n/2+1] to float32[n] (the input taken as the Hermitian
// half-spectrum, the imaginary parts of bins 0 and n/2 ignored, as numpy).
// kIfft and kIrfft scale by 1/n.
enum class FftType { kFft, kIfft, kRfft, kIrfft };
const char* FftTypeName(FftType type);
bool IsInverse(FftType type);
bool IsReal(FftType type);
// Elements per input / output row: n, or n/2+1 on the complex side of a real
// transform.
int64_t InputLength(FftType type, int64_t n);
int64_t OutputLength(FftType type, int64_t n);

// MLX's limits (fft.cpp 23-25) and radices (35-38, in order of preference).
inline constexpr int kMaxStockhamSize = 4096;
inline constexpr int kMaxRaderSize = 2048;
inline constexpr int kMaxBluesteinSize = 2048;
inline constexpr int kNumRadices = 9;
inline constexpr int kRadices[kNumRadices] = {13, 11, 8, 7, 6, 5, 4, 3, 2};
// The largest transform (the four-step passes are at most 4096 each).
inline constexpr int64_t kMaxFftSize = int64_t{kMaxStockhamSize} *
                                       kMaxStockhamSize;

// MLX's FFTPlan (fft.cpp 75-88).
struct FftPlan {
  int n = 0;
  // Steps per radix (kRadices order) of the Stockham decomposition.
  std::array<int, kNumRadices> stockham = {};
  // Steps per radix of the Rader decomposition of rader_n - 1.
  std::array<int, kNumRadices> rader = {};
  int rader_n = 1;  // 1: no Rader factor
  int bluestein_n = -1;
  // Several passes (four-step, or multi-upload Bluestein when bluestein_n
  // > 0).
  bool four_step = false;
  int n1 = 0;
  int n2 = 0;
};

// plan_fft (fft.cpp 118-192) for 1 <= n <= kMaxFftSize; Unimplemented
// beyond what the kernels cover (a power of two above 2^24, or a
// bluestein_n above 2^24, i.e. other n above 2^23).
absl::StatusOr<FftPlan> PlanFft(int64_t n);

// compute_elems_per_thread (fft.cpp 194-248), including MLX's hand-tuned
// sizes. Not defined for n == 1 (RunFft copies).
int ElemsPerThread(const FftPlan& plan);

// The dispatch of one FFT kernel launch (fft.cpp 647-690): `plan` is the
// plan of the launch's own length (n, or n1 / n2 of a four-step pass),
// `sequences` the number of transforms of that length.
struct FftLaunchGeometry {
  int fft_size = 0;         // bluestein_n or n
  int elems_per_thread = 0;
  int threads_per_fft = 0;  // threadgroup z
  int batch_per_group = 0;  // threadgroup y: transforms per threadgroup
  int tg_mem_size = 0;      // complex64s of threadgroup memory (<= 4096)
  int64_t groups = 0;       // threadgroups (x)
};
FftLaunchGeometry LaunchGeometry(const FftPlan& plan, bool real,
                                 bool four_step, int64_t sequences);

// The Rader and Bluestein constants MLX computes on the host for each call
// (compute_raders_constants, fft.cpp 281-320; compute_bluestein_constants,
// 323-370), here in double and rounded once to float: one blob for the
// device, the offsets (256-byte aligned) of each table in it. Empty for
// Stockham and power-of-two four-step plans.
struct FftConstants {
  // The host copy (a caller may release it once uploaded) and its size.
  std::vector<uint8_t> bytes;
  uint64_t size = 0;
  uint64_t w_q = 0;        // Bluestein: complex64[bluestein_n]
  uint64_t w_k = 0;        // Bluestein: complex64[n]
  uint64_t b_q = 0;        // Rader: complex64[rader_n - 1]
  uint64_t g_q = 0;        // Rader: int16[rader_n - 1]
  uint64_t g_minus_q = 0;  // Rader: int16[rader_n - 1]
};
FftConstants MakeFftConstants(const FftPlan& plan);

// RunFft processes the rows in chunks of at most max_chunk_elements
// complex64s of its widest buffer (n, or bluestein_n) and at least one row,
// so each launch stays short and the workspace bounded.
inline constexpr int64_t kMaxChunkElements = int64_t{1} << 24;
int64_t ChunkRows(const FftPlan& plan, int64_t rows,
                  int64_t max_chunk_elements = kMaxChunkElements);

// The workspace RunFft needs: a complex64 row of n per chunk row for the
// four-step, two of bluestein_n for the multi-upload Bluestein, else 0.
uint64_t FftWorkspaceBytes(const FftPlan& plan, int64_t rows,
                           int64_t max_chunk_elements = kMaxChunkElements);

// Helpers shared with the tests.
namespace internal {
// exp(-2 pi i k / n), with k reduced modulo n first.
std::complex<double> Twiddle(int64_t k, int64_t n);
// In-place forward (unnormalized) FFT of a power-of-two length, in double.
void Pow2FftDouble(std::vector<std::complex<double>>& x);
int PrimitiveRoot(int n);
}  // namespace internal

}  // namespace fft
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFT_FFT_PLAN_H_
