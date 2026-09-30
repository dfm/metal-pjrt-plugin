// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Microbenchmark for the CPU-side cost of a kernel dispatch through
// rt::Stream::Launch. Needs a Metal device; run under scripts/device_lock.py:
//   bazel build //bench:dispatch_bench
//   scripts/device_lock.py -- bazel-bin/bench/dispatch_bench
// Prints us/dispatch for encoding alone (Launch calls) and for the whole run
// including Synchronize, for independent and dependent (chained) launches.
#include <algorithm>
#include <chrono>
#include <mach/mach_time.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include <Metal/Metal.hpp>

namespace rt = metal_pjrt::rt;

namespace {

constexpr char kMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
kernel void add3(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* c [[buffer(2)]],
                 constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
  if (i < n) c[i] = a[i] + b[i];
}
)";

void Check(const absl::Status& s, const char* what) {
  if (!s.ok()) {
    std::fprintf(stderr, "%s: %s\n", what, s.ToString().c_str());
    std::exit(1);
  }
}

double MachNow() {
  static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
  return mach_absolute_time() * 1e-9 * tb.numer / tb.denom;
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  const int n_launch = argc > 1 ? std::atoi(argv[1]) : 20000;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
  auto dev_or = rt::Device::Create(0);
  Check(dev_or.status(), "Device::Create");
  std::unique_ptr<rt::Device> dev = *std::move(dev_or);
  auto k = dev->GetKernel(kMsl, "add3");
  Check(k.status(), "GetKernel");
  const rt::Kernel* kernel = *k;
  auto s_or = dev->CreateStream();
  Check(s_or.status(), "CreateStream");
  std::unique_ptr<rt::Stream> stream = *std::move(s_or);

  constexpr uint32_t kN = 256;
  // A pool of small buffers, as XLA's BFC hands out sub-ranges of one region;
  // plus some unrelated allocations so the address map is not trivial.
  std::vector<void*> bufs;
  std::vector<void*> filler;
  for (int i = 0; i < 64; ++i) {
    auto a = dev->Allocate(kN * sizeof(float));
    Check(a.status(), "Allocate");
    bufs.push_back(a->ptr);
    auto f = dev->Allocate(4096);
    Check(f.status(), "Allocate");
    filler.push_back(f->ptr);
  }
  const uint32_t n = kN;

  for (int mode = 0; mode < 2; ++mode) {
    const char* label = mode == 0 ? "independent" : "chained    ";
    for (int r = 0; r < reps; ++r) {
      double t0 = Now();
      for (int i = 0; i < n_launch; ++i) {
        void *a, *b, *c;
        if (mode == 0) {
          a = bufs[(3 * i) % 60];
          b = bufs[(3 * i + 1) % 60];
          c = bufs[60 + (i % 4)];
        } else {
          // Each launch reads the previous launch's output.
          a = bufs[i % 32];
          b = bufs[32 + ((i + 31) % 32)];
          c = bufs[32 + (i % 32)];
        }
        Check(stream->Launch(*kernel, rt::Dim3{1, 1, 1}, rt::Dim3{kN, 1, 1},
                             {rt::KernelArg::Buffer(a), rt::KernelArg::Buffer(b),
                              rt::KernelArg::Buffer(c),
                              rt::KernelArg::Bytes(&n, sizeof(n))}),
              "Launch");
      }
      double t1 = Now();
      Check(stream->Synchronize(), "Synchronize");
      double t2 = Now();
      std::printf(
          "%s rep %d: %d launches, encode %.2f us/dispatch, wall %.2f "
          "us/dispatch (sync tail %.1f ms)\n",
          label, r, n_launch, (t1 - t0) * 1e6 / n_launch,
          (t2 - t0) * 1e6 / n_launch, (t2 - t1) * 1e3);
    }
  }
  // While-loop shape: k dependent dispatches, then read a 4-byte predicate
  // back to the host and wait (what XLA's WhileThunk does per iteration).
  for (int k : {0, 1, 12, -1}) {
    // k == -1: one dispatch then Synchronize, no host copy.
    const bool copy = k >= 0;
    if (k < 0) k = 1;
    const int iters = std::max(1, n_launch / 20);
    for (int r = 0; r < reps; ++r) {
      uint32_t pred = 0;
      double t0 = Now();
      for (int it = 0; it < iters; ++it) {
        for (int j = 0; j < k; ++j) {
          int i = it * k + j;
          Check(stream->Launch(
                    *kernel, rt::Dim3{1, 1, 1}, rt::Dim3{kN, 1, 1},
                    {rt::KernelArg::Buffer(bufs[i % 32]),
                     rt::KernelArg::Buffer(bufs[32 + ((i + 31) % 32)]),
                     rt::KernelArg::Buffer(bufs[32 + (i % 32)]),
                     rt::KernelArg::Bytes(&n, sizeof(n))}),
                "Launch");
        }
        if (copy) {
          Check(stream->MemcpyDeviceToHost(&pred, bufs[32], sizeof(pred)),
                "MemcpyDeviceToHost");
        }
        Check(stream->Synchronize(), "Synchronize");
      }
      double t1 = Now();
      std::printf("loop k=%-2d %s rep %d: %d iterations, %.1f us/iteration\n",
                  k, copy ? "+copy" : "     ", r, iters,
                  (t1 - t0) * 1e6 / iters);
    }
  }
  // Raw metal-cpp round trip: one dispatch per command buffer, commit, wait.
  {
    MTL::CommandQueue* q = dev->mtl()->newCommandQueue();
    auto ref_a = *dev->Resolve(bufs[0]);
    auto ref_c = *dev->Resolve(bufs[33]);
    // Variant 4: created from a descriptor with
    // errorOptions = EncoderExecutionStatus (per-encoder error states).
    MTL::CommandBufferDescriptor* desc =
        MTL::CommandBufferDescriptor::alloc()->init();
    desc->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
    for (int variant = 0; variant < 5; ++variant) {
      const int iters = 100;
      double t0 = Now();
      double host_to_start = 0, gpu = 0, sched = 0, notify = 0;
      for (int it = 0; it < iters; ++it) {
        MTL::CommandBuffer* cb =
            variant == 2   ? q->commandBufferWithUnretainedReferences()
            : variant == 4 ? q->commandBuffer(desc)
                           : q->commandBuffer();
        if (variant >= 1) {
          MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
          e->setComputePipelineState(kernel->pso());
          e->setBuffer(ref_a.buffer, ref_a.offset, 0);
          e->setBuffer(ref_a.buffer, ref_a.offset, 1);
          e->setBuffer(ref_c.buffer, ref_c.offset, 2);
          e->setBytes(&n, sizeof(n), 3);
          e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(kN, 1, 1));
          e->endEncoding();
        }
        double c0 = Now();
        double m0 = MachNow();
        cb->commit();
        if (variant == 3) {
          while (cb->status() != MTL::CommandBufferStatusCompleted &&
                 cb->status() != MTL::CommandBufferStatusError) {
          }
        } else {
          cb->waitUntilCompleted();
        }
        double m1 = MachNow();
        double c1 = Now();
        sched += cb->GPUStartTime() - m0;
        notify += m1 - cb->GPUEndTime();
        host_to_start += c1 - c0;
        gpu += cb->GPUEndTime() - cb->GPUStartTime();
      }
      double t1 = Now();
      std::printf("raw cb %s: %.1f us/roundtrip (commit->complete %.1f us, "
                  "commit->gpu start %.1f us, gpu %.1f us, gpu end->host %.1f us)\n",
                  variant == 0   ? "empty          "
                  : variant == 1 ? "1 dispatch     "
                  : variant == 2 ? "1 disp unretain"
                  : variant == 3 ? "1 disp spin    "
                                 : "1 disp errinfo ",
                  (t1 - t0) * 1e6 / iters, host_to_start * 1e6 / iters,
                  sched * 1e6 / iters, gpu * 1e6 / iters, notify * 1e6 / iters);
    }
    desc->release();
    q->release();
  }
  for (void* p : bufs) Check(dev->Deallocate(p), "Deallocate");
  for (void* p : filler) Check(dev->Deallocate(p), "Deallocate");
  return 0;
}
