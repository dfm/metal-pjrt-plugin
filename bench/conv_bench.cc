// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// The convolution library (metal_pjrt/conv/conv.h) on the cnn benchmark's
// layers (bench/jax_bench.py "cnn fwd+bwd", batch 32, f32): the two forward
// convolutions, the input gradient of the second (input dilation 2,
// flipped kernel) and both weight gradients. Needs a Metal device; run under
// scripts/device_lock.py:
//   bazel build //bench:conv_bench
//   scripts/device_lock.py -- bazel-bin/bench/conv_bench [rounds] [f32|f16|bf16]
// Each round runs a burst of 30 back-to-back RunConv calls per case and
// prints one JSON line per case with the burst's wall time per call in ms
// (the GPU stays busy, so this is the kernel time plus a few us of
// encoding). bench/conv_bench_mlx.py times MLX's convolutions the same way.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/conv/conv.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace conv = metal_pjrt::conv;
namespace rt = metal_pjrt::rt;

namespace {

void Check(const absl::Status& s, const char* what) {
  if (!s.ok()) {
    std::fprintf(stderr, "%s: %s\n", what, s.ToString().c_str());
    std::exit(1);
  }
}

struct Case {
  const char* name;
  conv::ConvParams p;
};

std::vector<Case> Cases(conv::ConvType t) {
  std::vector<Case> cases;
  conv::ConvParams c1;  // [32,32,32,3] * [32,3,3,3], SAME
  c1.type = t;
  c1.n = 32, c1.h = 32, c1.w = 32, c1.c = 3;
  c1.o = 32, c1.kh = 3, c1.kw = 3, c1.out_h = 32, c1.out_w = 32;
  c1.pad_lo[0] = c1.pad_lo[1] = 1;
  cases.push_back({"conv1 fwd 3->32", c1});
  conv::ConvParams c2;  // [32,32,32,32] * [64,3,3,32], stride 2, SAME (0, 1)
  c2.type = t;
  c2.n = 32, c2.h = 32, c2.w = 32, c2.c = 32;
  c2.o = 64, c2.kh = 3, c2.kw = 3, c2.out_h = 16, c2.out_w = 16;
  c2.stride[0] = c2.stride[1] = 2;
  cases.push_back({"conv2 fwd 32->64 s2", c2});
  conv::ConvParams ig;  // cotangent [32,16,16,64] -> [32,32,32,32]
  ig.type = t;
  ig.n = 32, ig.h = 16, ig.w = 16, ig.c = 64;
  ig.o = 32, ig.kh = 3, ig.kw = 3, ig.out_h = 32, ig.out_w = 32;
  ig.pad_lo[0] = ig.pad_lo[1] = 2;
  ig.idil[0] = ig.idil[1] = 2;
  ig.flip = true;
  cases.push_back({"conv2 input grad", ig});
  conv::ConvParams wg1 = c1;
  wg1.kind = conv::ConvKind::kWeightGrad;
  cases.push_back({"conv1 weight grad", wg1});
  conv::ConvParams wg2 = c2;
  wg2.kind = conv::ConvKind::kWeightGrad;
  cases.push_back({"conv2 weight grad", wg2});
  return cases;
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;
  const std::string type = argc > 2 ? argv[2] : "f32";
  const conv::ConvType t = type == "f16"    ? conv::ConvType::kF16
                           : type == "bf16" ? conv::ConvType::kBF16
                                            : conv::ConvType::kF32;
  const int64_t item = t == conv::ConvType::kF32 ? 4 : 2;
  constexpr int kBurst = 30;
  auto dev_or = rt::Device::Create(0);
  Check(dev_or.status(), "device");
  std::unique_ptr<rt::Device> dev = *std::move(dev_or);
  auto stream_or = dev->CreateStream();
  Check(stream_or.status(), "stream");
  std::unique_ptr<rt::Stream> stream = *std::move(stream_or);

  for (const Case& c : Cases(t)) {
    const conv::ConvParams& p = c.p;
    absl::StatusOr<conv::ConvPlan> plan = conv::PlanConv(p);
    Check(plan.status(), c.name);
    auto alloc = [&](uint64_t bytes) {
      absl::StatusOr<rt::Allocation> a = dev->Allocate(std::max<uint64_t>(bytes, 16));
      Check(a.status(), "allocate");
      std::memset(a->ptr, 0, bytes);
      return a->ptr;
    };
    // Forward: in, weight -> out; weight gradient: in, dY -> dW.
    const bool wgrad = p.kind == conv::ConvKind::kWeightGrad;
    const int64_t wt_bytes = p.o * p.kh * p.kw * p.c * item;
    const int64_t out_bytes = p.n * p.out_h * p.out_w * p.o * item;
    void* in = alloc(p.n * p.h * p.w * p.c * item);
    void* wt = alloc(wgrad ? out_bytes : wt_bytes);
    void* out = alloc(wgrad ? wt_bytes : out_bytes);
    void* ws = plan->workspace_bytes ? alloc(plan->workspace_bytes) : nullptr;
    for (int i = 0; i < 3; ++i) {  // compile + warm up
      Check(conv::RunConv(dev.get(), stream.get(), p, *plan, in, wt, out, ws),
            "warmup");
    }
    Check(stream->Synchronize(), "sync");
    for (int r = 0; r < rounds; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kBurst; ++i) {
        Check(conv::RunConv(dev.get(), stream.get(), p, *plan, in, wt, out,
                            ws),
              "run");
      }
      Check(stream->Synchronize(), "sync");
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count() /
                        kBurst;
      std::printf(
          "{\"backend\": \"metal-conv-lib\", \"case\": \"%s\", \"type\": "
          "\"%s\", \"path\": \"%s\", \"ms\": %.4f, \"gflops\": %.1f}\n",
          c.name, type.c_str(), conv::ConvPathName(plan->path), ms,
          conv::ConvFlops(p) / (ms * 1e6));
    }
    for (void* ptr : {in, wt, out, ws}) {
      if (ptr != nullptr) Check(dev->Deallocate(ptr), "free");
    }
  }
  return 0;
}
