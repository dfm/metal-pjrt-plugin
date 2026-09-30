// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Small-M half-precision GEMMs, D[M, N] = x[M, K] W^T (default N = 6144,
// K = 1024), on the kernels MetalBlasLt picks (the wide gemv for 2..8
// rows, else steel) and on steel with a few tiles, cycling through up to 28
// weights (336 MB at the default shape, so that little stays in the system
// cache: the LLM decode case); "nn" stores W as [K, N] instead (x W).
// Needs a Metal device:
//   bazel build //bench:gemm_bench
//   scripts/device_lock.py -- bazel-bin/bench/gemm_bench \
//       [rounds] [bf16|f16] [N] [K] [nt|nn]
// Each round runs a burst of 4 passes over the weights per case and prints
// one JSON line with the wall time per GEMM in ms and the weight bandwidth
// (the GPU stays busy, so this is the kernel time plus a few us of
// encoding).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/blas/gemv.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace blas = metal_pjrt::blas;
namespace rt = metal_pjrt::rt;

namespace {

void Check(const absl::Status& s, const char* what) {
  if (!s.ok()) {
    std::fprintf(stderr, "%s: %s\n", what, s.ToString().c_str());
    std::exit(1);
  }
}


}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;
  const std::string type = argc > 2 ? argv[2] : "bf16";
  const int64_t kN = argc > 3 ? std::atoll(argv[3]) : 6144;
  const int64_t kK = argc > 4 ? std::atoll(argv[4]) : 1024;
  // "nt": x W^T with W stored [N, K]; "nn": x W with W stored [K, N].
  const bool nt = argc <= 5 || std::string(argv[5]) != "nn";
  // Up to 28 weights, at most ~400 MB of them.
  const int64_t kWeights =
      std::max<int64_t>(1, std::min<int64_t>(28, (400 << 20) / (kN * kK * 2)));
  const blas::MpsDType t =
      type == "f16" ? blas::MpsDType::kF16 : blas::MpsDType::kBF16;
  auto dev_or = rt::Device::Create(0);
  Check(dev_or.status(), "device");
  std::unique_ptr<rt::Device> dev = *std::move(dev_or);
  auto stream_or = dev->CreateStream();
  Check(stream_or.status(), "stream");
  std::unique_ptr<rt::Stream> stream = *std::move(stream_or);

  auto operand = [&](int64_t rows, int64_t cols, bool trans) {
    absl::StatusOr<rt::Allocation> a = dev->Allocate(rows * cols * 2);
    Check(a.status(), "allocate");
    std::memset(a->ptr, 0, rows * cols * 2);
    absl::StatusOr<rt::BufferRef> ref = dev->Resolve(a->ptr);
    Check(ref.status(), "resolve");
    blas::MpsOperand o;
    o.buffer = ref->buffer;
    o.offset = ref->offset;
    o.ld = cols;
    o.dtype = t;
    o.transpose = trans;
    return o;
  };
  std::vector<blas::MpsOperand> weights;
  for (int i = 0; i < kWeights; ++i) {
    weights.push_back(nt ? operand(kN, kK, true) : operand(kK, kN, false));
  }
  const blas::MpsOperand x = operand(64, kK, false);
  const blas::MpsOperand d = operand(64, kN, false);

  struct Tile {
    const char* name;
    std::optional<blas::SteelTile> tile;  // nullopt: MetalBlasLt's choice
  };
  using ST = blas::SteelTile;
  const Tile tiles[] = {{"auto", std::nullopt},
                        {"steel 64x32x32 2x2", ST{64, 32, 32, 2, 2}},
                        {"steel 32x32x16 2x2", ST{32, 32, 16, 2, 2}},
                        {"steel 16x64x16 1x2", ST{16, 64, 16, 1, 2}},
                        {"steel 16x32x32 1x2", ST{16, 32, 32, 1, 2}},
                        {"steel 32x64x16 1x2", ST{32, 64, 16, 1, 2}}};
  constexpr int kReps = 4;
  for (int64_t m : {1, 2, 4, 8, 12, 15, 16, 24, 32, 48, 64}) {
    for (const Tile& tile : tiles) {
      auto run = [&](const blas::MpsOperand& w) {
        blas::GemmParams p;
        p.m = m;
        p.n = kN;
        p.k = kK;
        p.a = x;
        p.b = w;
        p.c = d;
        p.c.ld = kN;
        p.c.transpose = false;
        if (!tile.tile.has_value()) {
          if (auto plan = blas::ChooseGemv(p, dev->info().gpu_family)) {
            return blas::RunGemv(dev.get(), stream.get(), p, *plan);
          }
          return blas::RunSteelGemm(dev.get(), stream.get(), p);
        }
        return blas::RunSteelGemm(dev.get(), stream.get(), p, &*tile.tile);
      };
      for (int i = 0; i < kWeights; ++i) Check(run(weights[i]), "warmup");
      Check(stream->Synchronize(), "sync");
      for (int r = 0; r < rounds; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < kReps; ++rep) {
          for (int i = 0; i < kWeights; ++i) Check(run(weights[i]), "run");
        }
        Check(stream->Synchronize(), "sync");
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count() /
                          (kReps * kWeights);
        std::printf(
            "{\"backend\": \"metal-gemm-lib\", \"type\": \"%s\", \"n\": %lld, "
            "\"k\": %lld, \"layout\": \"%s\", \"m\": %lld, "
            "\"kernel\": \"%s\", \"ms\": %.4f, \"GB/s\": %.1f}\n",
            type.c_str(), static_cast<long long>(kN),
            static_cast<long long>(kK), nt ? "nt" : "nn",
            static_cast<long long>(m), tile.name, ms,
            kN * kK * 2 / (ms * 1e6));
      }
    }
  }
  return 0;
}
