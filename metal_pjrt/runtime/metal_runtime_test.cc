// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Tests for the runtime layer on its own (no XLA). Needs a Metal device.
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "metal_pjrt/runtime/system_memory.h"
#include "metal_pjrt/kernels/msl_prelude.metal.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace metal_pjrt {
namespace rt {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr char kMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; float alpha; };
kernel void axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 constant Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) y[i] = p.alpha * x[i] + y[i];
}
)";

struct Params {
  uint32_t n;
  float alpha;
};

class MetalRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<Device>> dev = Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<const Kernel*> k = dev_->GetKernel(kMsl, "axpy");
    ASSERT_THAT(k, IsOk());
    kernel_ = *k;
  }

  void* Alloc(uint64_t size) {
    absl::StatusOr<Allocation> a = dev_->Allocate(size);
    EXPECT_THAT(a, IsOk());
    return a.ok() ? a->ptr : nullptr;
  }

  std::unique_ptr<Stream> NewStream() {
    absl::StatusOr<std::unique_ptr<Stream>> s = dev_->CreateStream();
    EXPECT_THAT(s, IsOk());
    return s.ok() ? *std::move(s) : nullptr;
  }

  absl::Status Axpy(Stream* s, void* x, void* y, uint32_t n, float alpha,
                    uint32_t groups) {
    Params p{n, alpha};
    return s->Launch(*kernel_, Dim3{groups, 1, 1}, Dim3{256, 1, 1},
                     {KernelArg::Buffer(x), KernelArg::Buffer(y),
                      KernelArg::Bytes(&p, sizeof(p))});
  }

  std::unique_ptr<Device> dev_;
  const Kernel* kernel_ = nullptr;
};

TEST_F(MetalRuntimeTest, DeviceInfo) {
  EXPECT_GE(Device::VisibleDeviceCount(), 1);
  EXPECT_FALSE(dev_->info().name.empty());
  EXPECT_GT(dev_->info().max_buffer_length, 0u);
  EXPECT_EQ(kernel_->thread_execution_width(), 32u);
}

TEST_F(MetalRuntimeTest, KernelCache) {
  absl::StatusOr<const Kernel*> again = dev_->GetKernel(kMsl, "axpy");
  ASSERT_THAT(again, IsOk());
  EXPECT_EQ(*again, kernel_);
  EXPECT_FALSE(kernel_->uses_argument_buffer());
}

// Kernels taken with AcquireKernel (compiled executables' kernels) go with
// their last reference, and their source's library and text with its last
// kernel; GetKernel's kernels stay. A kernel released while its launch is in
// flight still runs (the command buffer retains the pipeline).
TEST_F(MetalRuntimeTest, AcquiredKernelsGoWithTheirLastReference) {
  const Device::KernelCacheStats base = dev_->kernel_cache_stats();
  const std::string msl =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void one(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) { y[i] = 1; }\n"
      "kernel void two(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) { y[i] = 2; }\n";
  absl::StatusOr<const Kernel*> a = dev_->AcquireKernel(msl, "one");
  absl::StatusOr<const Kernel*> b = dev_->AcquireKernel(msl, "one");
  absl::StatusOr<const Kernel*> c = dev_->AcquireKernel(msl, "two");
  ASSERT_THAT(a, IsOk());
  ASSERT_THAT(b, IsOk());
  ASSERT_THAT(c, IsOk());
  EXPECT_EQ(*a, *b);
  Device::KernelCacheStats s = dev_->kernel_cache_stats();
  EXPECT_EQ(s.libraries, base.libraries + 1);
  EXPECT_EQ(s.kernels, base.kernels + 2);
  EXPECT_EQ(s.source_bytes, base.source_bytes + msl.size());

  std::unique_ptr<Stream> stream = NewStream();
  auto* y = static_cast<int32_t*>(Alloc(1024 * sizeof(int32_t)));
  ASSERT_THAT(stream->Launch(**c, Dim3{4, 1, 1}, Dim3{256, 1, 1},
                             {KernelArg::Buffer(y)}),
              IsOk());
  dev_->ReleaseKernel(*c);  // "two" goes, the library stays for "one"
  s = dev_->kernel_cache_stats();
  EXPECT_EQ(s.libraries, base.libraries + 1);
  EXPECT_EQ(s.kernels, base.kernels + 1);
  ASSERT_THAT(stream->Synchronize(), IsOk());
  for (int i = 0; i < 1024; ++i) ASSERT_EQ(y[i], 2) << i;

  dev_->ReleaseKernel(*a);
  EXPECT_EQ(dev_->kernel_cache_stats().kernels, base.kernels + 1);
  dev_->ReleaseKernel(*b);
  s = dev_->kernel_cache_stats();
  EXPECT_EQ(s.libraries, base.libraries);
  EXPECT_EQ(s.kernels, base.kernels);
  EXPECT_EQ(s.source_bytes, base.source_bytes);

  // Acquired and got: GetKernel's pin keeps it after the last release.
  absl::StatusOr<const Kernel*> d = dev_->AcquireKernel(msl, "one");
  ASSERT_THAT(d, IsOk());
  ASSERT_THAT(dev_->GetKernel(msl, "one"), IsOk());
  EXPECT_EQ(*dev_->GetKernel(msl, "one"), *d);
  dev_->ReleaseKernel(*d);
  EXPECT_EQ(dev_->kernel_cache_stats().kernels, base.kernels + 1);
  ASSERT_THAT(stream->Launch(**d, Dim3{4, 1, 1}, Dim3{256, 1, 1},
                             {KernelArg::Buffer(y)}),
              IsOk());
  ASSERT_THAT(stream->Synchronize(), IsOk());
  for (int i = 0; i < 1024; ++i) ASSERT_EQ(y[i], 1) << i;
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// Threads acquiring and releasing kernels of one shared source (the library
// dropped and recompiled as its last kernel comes and goes, other threads'
// acquires waiting on it) and of their own: every acquire gets the kernel
// asked for, and the cache ends where it started.
TEST_F(MetalRuntimeTest, ConcurrentAcquireAndRelease) {
  const Device::KernelCacheStats base = dev_->kernel_cache_stats();
  const std::string shared =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void one(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) { y[i] = 1; }\n"
      "kernel void two(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) { y[i] = 2; }\n";
  constexpr int kThreads = 4;
  constexpr int kIterations = 12;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const std::string own = absl::StrCat(
          "#include <metal_stdlib>\nusing namespace metal;\n"
          "kernel void mine(device int* y [[buffer(0)]],\n"
          "    uint i [[thread_position_in_grid]]) { y[i] = ",
          t, "; }\n");
      for (int i = 0; i < kIterations; ++i) {
        const char* name = (i + t) % 2 == 0 ? "one" : "two";
        absl::StatusOr<const Kernel*> a = dev_->AcquireKernel(shared, name);
        absl::StatusOr<const Kernel*> b = dev_->AcquireKernel(own, "mine");
        if (!a.ok() || !b.ok() || (*a)->name() != name ||
            (*b)->name() != "mine") {
          ++failures;
        }
        if (a.ok()) dev_->ReleaseKernel(*a);
        if (b.ok()) dev_->ReleaseKernel(*b);
      }
    });
  }
  for (std::thread& t : threads) t.join();
  EXPECT_EQ(failures.load(), 0);
  const Device::KernelCacheStats s = dev_->kernel_cache_stats();
  EXPECT_EQ(s.libraries, base.libraries);
  EXPECT_EQ(s.kernels, base.kernels);
  EXPECT_EQ(s.source_bytes, base.source_bytes);
}

// Releasing a library's last kernel while a launch of it is queued drops the
// library from the cache; the command buffer keeps the pipeline, so the
// launch still runs.
TEST_F(MetalRuntimeTest, LibraryDroppedWhileLaunchQueued) {
  const Device::KernelCacheStats base = dev_->kernel_cache_stats();
  const std::string msl =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void slow(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  int v = 0;\n"
      "  for (int k = 0; k < 4096; ++k) v += (int(i) + k) & 1;\n"
      "  y[i] = v;\n}\n";
  absl::StatusOr<const Kernel*> k = dev_->AcquireKernel(msl, "slow");
  ASSERT_THAT(k, IsOk());
  EXPECT_EQ(dev_->kernel_cache_stats().libraries, base.libraries + 1);
  constexpr int kN = 1 << 18;
  std::unique_ptr<Stream> stream = NewStream();
  auto* y = static_cast<int32_t*>(Alloc(kN * sizeof(int32_t)));
  ASSERT_THAT(stream->Launch(**k, Dim3{kN / 256, 1, 1}, Dim3{256, 1, 1},
                             {KernelArg::Buffer(y)}),
              IsOk());
  dev_->ReleaseKernel(*k);
  const Device::KernelCacheStats s = dev_->kernel_cache_stats();
  EXPECT_EQ(s.libraries, base.libraries);
  EXPECT_EQ(s.kernels, base.kernels);
  ASSERT_THAT(stream->Synchronize(), IsOk());
  for (int i = 0; i < kN; ++i) ASSERT_EQ(y[i], 2048) << i;
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A source starting with kMslPreludeLine compiles with the MSL prelude in
// its place; the cache keeps the short text, the identity hashes the full
// one (the same key as the source written out in full).
TEST_F(MetalRuntimeTest, PreludeLine) {
  const std::string body =
      "\nkernel void pre(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  y[i] = xla_vext<int>(int2(3, 4), 1);\n}\n";
  const std::string msl = kMslPreludeLine + body;
  const Device::KernelCacheStats base = dev_->kernel_cache_stats();
  absl::StatusOr<const Kernel*> k = dev_->AcquireKernel(msl, "pre");
  ASSERT_THAT(k, IsOk());
  EXPECT_EQ(dev_->kernel_cache_stats().source_bytes,
            base.source_bytes + msl.size());
  absl::StatusOr<const Kernel*> full = dev_->AcquireKernel(
      std::string(kernels::kMslPrelude) + body, "pre");
  ASSERT_THAT(full, IsOk());
  EXPECT_NE(*k, *full);
  EXPECT_EQ((*k)->key(), (*full)->key());

  std::unique_ptr<Stream> s = NewStream();
  auto* y = static_cast<int32_t*>(Alloc(64 * sizeof(int32_t)));
  ASSERT_THAT(s->Launch(**k, Dim3{1, 1, 1}, Dim3{64, 1, 1},
                        {KernelArg::Buffer(y)}),
              IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  for (int i = 0; i < 64; ++i) ASSERT_EQ(y[i], 4) << i;
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
  dev_->ReleaseKernel(*k);
  dev_->ReleaseKernel(*full);
  EXPECT_EQ(dev_->kernel_cache_stats().kernels, base.kernels);
}

// Function constants select a variant at pipeline creation: one kernel per
// (source, function, constants), each with its own identity key.
TEST_F(MetalRuntimeTest, FunctionConstants) {
  const std::string msl =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "constant bool kNegate [[function_constant(0)]];\n"
      "constant int kAdd [[function_constant(1)]];\n"
      "kernel void variant(device int* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) {\n"
      "  int v = int(i) + kAdd;\n"
      "  y[i] = kNegate ? -v : v;\n}\n";
  const FunctionConstant a[] = {FunctionConstant::Bool(0, false),
                                FunctionConstant::Int(1, 10)};
  const FunctionConstant b[] = {FunctionConstant::Bool(0, true),
                                FunctionConstant::Int(1, 10)};
  absl::StatusOr<const Kernel*> ka = dev_->GetKernel(msl, "variant", a);
  absl::StatusOr<const Kernel*> kb = dev_->GetKernel(msl, "variant", b);
  ASSERT_THAT(ka, IsOk());
  ASSERT_THAT(kb, IsOk());
  EXPECT_NE(*ka, *kb);
  EXPECT_NE((*ka)->key(), (*kb)->key());
  EXPECT_THAT(dev_->GetKernel(msl, "variant", a), IsOk());
  EXPECT_EQ(*dev_->GetKernel(msl, "variant", a), *ka);

  std::unique_ptr<Stream> s = NewStream();
  auto* y = static_cast<int32_t*>(Alloc(8 * sizeof(int32_t)));
  for (const Kernel* k : {*ka, *kb}) {
    ASSERT_THAT(s->Launch(*k, Dim3{1, 1, 1}, Dim3{8, 1, 1},
                          {KernelArg::Buffer(y)}),
                IsOk());
    ASSERT_THAT(s->Synchronize(), IsOk());
    const int sign = k == *ka ? 1 : -1;
    for (int i = 0; i < 8; ++i) ASSERT_EQ(y[i], sign * (i + 10)) << i;
  }
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

TEST_F(MetalRuntimeTest, LaunchCopyAndEvents) {
  const uint32_t n = 1 << 20;
  const uint32_t groups = (n + 255) / 256;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  void* z = Alloc(n * 4);
  absl::StatusOr<BufferRef> ref = dev_->Resolve(static_cast<char*>(y) + 4096);
  ASSERT_THAT(ref, IsOk());
  EXPECT_EQ(ref->offset, 4096u);

  std::unique_ptr<Stream> s = NewStream();
  std::vector<float> hx(n, 1.0f), hy(n, 2.0f), out(n, 0.0f);
  ASSERT_THAT(s->MemcpyHostToDevice(x, hx.data(), n * 4), IsOk());
  ASSERT_THAT(s->MemcpyHostToDevice(y, hy.data(), n * 4), IsOk());
  // y += 3x ten times -> y = 2 + 30 = 32.
  for (int i = 0; i < 10; ++i) ASSERT_THAT(Axpy(s.get(), x, y, n, 3.0f, groups), IsOk());
  // D2D copy then D2H, all ordered on the stream.
  ASSERT_THAT(s->MemcpyDeviceToDevice(z, y, n * 4), IsOk());
  ASSERT_THAT(s->Memset8(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->MemcpyDeviceToHost(out.data(), z, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  int bad = 0;
  for (uint32_t i = 0; i < n; ++i) bad += out[i] != 32.0f;
  EXPECT_EQ(bad, 0);
  EXPECT_EQ(static_cast<float*>(y)[123], 0.0f);

  // Events across two streams: s2 waits for s's work.
  std::unique_ptr<Stream> s2 = NewStream();
  absl::StatusOr<std::unique_ptr<Event>> ev = dev_->CreateEvent();
  ASSERT_THAT(ev, IsOk());
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, groups), IsOk());  // y = 1
  ASSERT_THAT(s->RecordEvent(ev->get()), IsOk());
  ASSERT_THAT(s2->WaitForEvent(ev->get()), IsOk());
  ASSERT_THAT(Axpy(s2.get(), x, y, n, 1.0f, groups), IsOk());  // y = 2
  ASSERT_THAT(s2->Synchronize(), IsOk());
  EXPECT_TRUE((*ev)->IsComplete());
  EXPECT_EQ(static_cast<float*>(y)[777], 2.0f);

  // Host callback ordering + Memset32.
  int calls = 0;
  ASSERT_THAT(s->HostCallback([&calls]() {
    ++calls;
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(s->Memset32(z, 0x3f800000u, n * 4), IsOk());  // 1.0f pattern
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(static_cast<float*>(z)[5], 1.0f);

  // Many small launches exercise command-buffer rollover (> 64 ops).
  for (int i = 0; i < 200; ++i) ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[0], 202.0f);

  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
  EXPECT_THAT(dev_->Deallocate(z), IsOk());
  EXPECT_EQ(dev_->allocated_bytes(), 0u);
}

TEST_F(MetalRuntimeTest, FillPatterns) {
  const uint32_t n = 4096;
  void* z = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  // Non-uniform 32-bit pattern goes through the fill kernel.
  ASSERT_THAT(s->Memset32(z, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(z)[0], 1.0f);
  EXPECT_EQ(static_cast<float*>(z)[n - 1], 1.0f);
  // Byte fill of an unaligned interior range: blit path.
  ASSERT_THAT(s->Memset8(static_cast<char*>(z) + 3, 0xab, 10), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  const uint8_t* bytes = static_cast<const uint8_t*>(z);
  // 1.0f is 00 00 80 3f in memory.
  EXPECT_EQ(bytes[2], 0x80);
  EXPECT_EQ(bytes[3], 0xab);
  EXPECT_EQ(bytes[12], 0xab);
  EXPECT_EQ(bytes[13], 0x00);
  EXPECT_EQ(bytes[15], 0x3f);
  EXPECT_THAT(dev_->Deallocate(z), IsOk());
}

TEST_F(MetalRuntimeTest, ResetLogAndQuarantine) {
  // A private state directory so the real reset log is untouched.
  char tmpl[] = "/tmp/metal_rt_state_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string dir = tmpl;
  setenv("METAL_PJRT_STATE_DIR", dir.c_str(), 1);
  // Opt-in (off by default, below).
  setenv("METAL_PJRT_QUARANTINE_STRIKES", "2", 1);
  absl::StatusOr<std::unique_ptr<Device>> d1 = Device::Create(0);
  ASSERT_THAT(d1, IsOk());
  EXPECT_EQ((*d1)->state_dir(), dir);
  EXPECT_EQ((*d1)->resets_since_boot(), 0);
  absl::StatusOr<const Kernel*> k1 = (*d1)->GetKernel(kMsl, "axpy");
  ASSERT_THAT(k1, IsOk());
  EXPECT_THAT((*k1)->key(), HasSubstr(":axpy"));

  // Two resets blaming axpy (the second with a duplicate and a null entry).
  (*d1)->RecordReset("first \"timeout\"", {(*k1)->identity()});
  EXPECT_EQ((*d1)->resets_since_boot(), 1);
  (*d1)->RecordReset("second", {(*k1)->identity(), (*k1)->identity(), nullptr});
  std::ifstream in(dir + "/gpu_resets.jsonl");
  std::string line;
  int lines = 0;
  while (std::getline(in, line)) {
    ++lines;
    EXPECT_THAT(line, HasSubstr("\"key\":\"" + (*k1)->key() + "\""));
    EXPECT_THAT(line, HasSubstr("\"name\":\"axpy\""));
  }
  EXPECT_EQ(lines, 2);

  // A new device (new process, effectively) refuses the kernel.
  absl::StatusOr<std::unique_ptr<Device>> d2 = Device::Create(0);
  ASSERT_THAT(d2, IsOk());
  EXPECT_EQ((*d2)->resets_since_boot(), 2);
  // The remedy works without the source checkout (gpu_health.py).
  EXPECT_THAT((*d2)->GetKernel(kMsl, "axpy"),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("quarantined"),
                             HasSubstr("deleting " + dir + "/gpu_resets.jsonl"))));

  // Built-in kernels are listed by name, never struck or refused.
  absl::StatusOr<const Kernel*> fill =
      (*d2)->BuiltinKernel(Device::Builtin::kFill32);
  ASSERT_THAT(fill, IsOk());
  (*d2)->RecordReset("builtin", {(*fill)->identity()});
  (*d2)->RecordReset("builtin", {(*fill)->identity()});
  {
    std::ifstream in2(dir + "/gpu_resets.jsonl");
    std::string last;
    while (std::getline(in2, line)) last = line;
    EXPECT_THAT(last, HasSubstr("\"builtins\":[\"xla_metal_fill32\"]"));
    EXPECT_THAT(last, HasSubstr("\"kernels\":[]"));
    EXPECT_THAT(last, HasSubstr("\"build\":\""));
  }
  absl::StatusOr<std::unique_ptr<Device>> d2b = Device::Create(0);
  ASSERT_THAT(d2b, IsOk());
  EXPECT_THAT((*d2b)->BuiltinKernel(Device::Builtin::kFill32), IsOk());

  // Strikes count only since boot, from any plugin build: rewrite the log
  // with the two axpy resets from before boot plus one since boot from
  // another build (one strike), then add a second from another build.
  std::vector<std::string> records;
  {
    std::ifstream in3(dir + "/gpu_resets.jsonl");
    while (std::getline(in3, line)) records.push_back(line);
  }
  auto other_build = [](std::string l) {
    size_t b = l.find("\"build\":\"") + 9;
    l.replace(b, l.find('"', b) - b, "0123");
    return l;
  };
  {
    std::ofstream out(dir + "/gpu_resets.jsonl", std::ios::trunc);
    for (int i = 0; i < 2; ++i) {
      std::string old = records[i];
      old.replace(8, old.find(',') - 8, "1");  // "time":1
      out << old << "\n";
    }
    out << other_build(records[0]) << "\n";
  }
  absl::StatusOr<std::unique_ptr<Device>> d4 = Device::Create(0);
  ASSERT_THAT(d4, IsOk());
  EXPECT_EQ((*d4)->resets_since_boot(), 1);
  EXPECT_THAT((*d4)->GetKernel(kMsl, "axpy"), IsOk());
  {
    std::ofstream out(dir + "/gpu_resets.jsonl", std::ios::app);
    out << other_build(records[1]) << "\n";
  }
  absl::StatusOr<std::unique_ptr<Device>> d5 = Device::Create(0);
  ASSERT_THAT(d5, IsOk());
  EXPECT_THAT((*d5)->GetKernel(kMsl, "axpy"),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Disabled by threshold 0, and by default: the kernel runs again.
  setenv("METAL_PJRT_QUARANTINE_STRIKES", "0", 1);
  absl::StatusOr<std::unique_ptr<Device>> d3 = Device::Create(0);
  ASSERT_THAT(d3, IsOk());
  EXPECT_THAT((*d3)->GetKernel(kMsl, "axpy"), IsOk());
  unsetenv("METAL_PJRT_QUARANTINE_STRIKES");
  absl::StatusOr<std::unique_ptr<Device>> d6 = Device::Create(0);
  ASSERT_THAT(d6, IsOk());
  EXPECT_EQ((*d6)->quarantine_strikes(), 0);
  EXPECT_EQ((*d6)->resets_since_boot(), 2);  // still logged and counted
  EXPECT_THAT((*d6)->GetKernel(kMsl, "axpy"), IsOk());
  unsetenv("METAL_PJRT_STATE_DIR");
  std::filesystem::remove_all(dir);

  // Without HOME the log goes under the account's home, never the working
  // directory.
  const char* home = std::getenv("HOME");
  const std::string saved = home != nullptr ? home : "";
  unsetenv("HOME");
  absl::StatusOr<std::unique_ptr<Device>> d7 = Device::Create(0);
  if (!saved.empty()) setenv("HOME", saved.c_str(), 1);
  ASSERT_THAT(d7, IsOk());
  EXPECT_EQ((*d7)->state_dir().rfind("/", 0), 0u) << (*d7)->state_dir();
  EXPECT_THAT((*d7)->state_dir(), HasSubstr("/.cache/metal-pjrt"));
}

// Waits across streams: Synchronize, events and host tasks of a stream that
// waits for another honor that stream's work.
TEST_F(MetalRuntimeTest, WaitsAcrossStreams) {
  const uint32_t n = 1 << 18;
  const uint32_t groups = (n + 255) / 256;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  std::unique_ptr<Stream> s2 = NewStream();
  std::vector<float> hx(n, 1.0f), hy(n, 0.0f), out(n, -1.0f);
  ASSERT_THAT(s->MemcpyHostToDevice(x, hx.data(), n * 4), IsOk());
  ASSERT_THAT(s->MemcpyHostToDevice(y, hy.data(), n * 4), IsOk());

  // s does a lot of work; s2 waits for it and then syncs with no GPU work of
  // its own: the wait must be honored on the host.
  for (int i = 0; i < 300; ++i) ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, groups), IsOk());
  absl::StatusOr<std::unique_ptr<Event>> ev = dev_->CreateEvent();
  ASSERT_THAT(ev, IsOk());
  ASSERT_THAT(s->RecordEvent(ev->get()), IsOk());
  ASSERT_THAT(s2->WaitForEvent(ev->get()), IsOk());
  ASSERT_THAT(s2->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[n - 1], 300.0f);

  // XLA's device-to-host pattern: wait for the compute stream, then copy on
  // the transfer stream (a host task), then block. The task must wait for
  // the compute stream's work, which s2 never encoded anything behind.
  for (int i = 0; i < 300; ++i) ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, groups), IsOk());
  ASSERT_THAT(s2->WaitForStream(s.get()), IsOk());
  ASSERT_THAT(s2->MemcpyDeviceToHost(out.data(), y, n * 4), IsOk());
  ASSERT_THAT(s2->Synchronize(), IsOk());
  EXPECT_EQ(out[0], 600.0f);
  EXPECT_EQ(out[n - 1], 600.0f);

  // A host task followed by GPU work on the same stream: the work waits for
  // the task.
  std::atomic<int> ran{0};
  ASSERT_THAT(s->HostCallback([&]() {
    static_cast<float*>(y)[0] = 1000.0f;
    ran = 1;
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, groups), IsOk());  // y[0] = 1001
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(ran.load(), 1);
  EXPECT_EQ(static_cast<float*>(y)[0], 1001.0f);
  EXPECT_EQ(static_cast<float*>(y)[1], 601.0f);

  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// GPU work behind a slow host task is encoded only after the task has run,
// so no command buffer waits on host work on the GPU.
TEST_F(MetalRuntimeTest, GpuWorkAfterAHostTaskWaitsOnTheHost) {
  const uint32_t n = 1 << 16;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  const uint64_t holds = dev_->encode_host_waits();
  ASSERT_THAT(s->HostCallback([y]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    static_cast<float*>(y)[0] = 100.0f;
    return absl::OkStatus();
  }), IsOk());
  // Encoded only once the task has run.
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(s->Flush(), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[0], 101.0f);
  EXPECT_EQ(static_cast<float*>(y)[n - 1], 1.0f);
  EXPECT_GT(dev_->encode_host_waits(), holds);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// Work of another stream behind a backlog never waits on the GPU: an event
// recorded after only waits (XLA: a zero-byte device-to-device copy behind
// the compute sync point) needs no command buffer at all, and a copy behind
// the backlog is simply later in the one queue.
TEST_F(MetalRuntimeTest, WorkBehindABacklogNeverWaitsOnTheGpu) {
  const uint32_t n = 1 << 20;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  void* z = Alloc(n * 4);
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  absl::StatusOr<std::unique_ptr<Event>> e = dev_->CreateEvent();
  ASSERT_THAT(e, IsOk());
  ASSERT_THAT(a->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(a->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(a->Synchronize(), IsOk());
  // A backlog of short kernels (~0.1 ms each; the thread cap commits a
  // buffer every 128 of them).
  const int kLaunches = 1000;
  for (int i = 0; i < kLaunches; ++i) {
    ASSERT_THAT(Axpy(a.get(), x, y, n, 1.0f, n / 256), IsOk());
  }
  const uint64_t committed_before = a->FenceForTesting().first;
  ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
  ASSERT_THAT(b->RecordEvent(e->get()), IsOk());  // commits a's open buffer
  EXPECT_LE(b->FenceForTesting().first, committed_before + 1);
  ASSERT_THAT(b->MemcpyDeviceToDevice(z, y, n * 4), IsOk());
  ASSERT_THAT((*e)->WaitOnHost(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[0], static_cast<float>(kLaunches));
  ASSERT_THAT(b->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(z)[0], static_cast<float>(kLaunches));
  EXPECT_EQ(static_cast<float*>(z)[n - 1], static_cast<float>(kLaunches));
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
  EXPECT_THAT(dev_->Deallocate(z), IsOk());
}

// Freeing anything but the start of a live allocation, with its size, is a
// plugin bug: INTERNAL, and nothing is cached or released.
TEST_F(MetalRuntimeTest, DeallocateRefusesBadPointers) {
  absl::StatusOr<Allocation> a = dev_->Allocate(5000);
  ASSERT_THAT(a, IsOk());
  const Device::MemoryStats before = dev_->memory_stats();
  char* p = static_cast<char*>(a->ptr);
  EXPECT_THAT(dev_->Deallocate(p + 16),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not the start of a live allocation")));
  EXPECT_THAT(dev_->Deallocate(p, 4096),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("the allocation there has 5000 bytes")));
  int unknown = 0;
  EXPECT_THAT(dev_->Deallocate(&unknown),
              StatusIs(absl::StatusCode::kInternal));
  const Device::MemoryStats same = dev_->memory_stats();
  EXPECT_EQ(same.live_bytes, before.live_bytes);
  EXPECT_EQ(same.cached_bytes, before.cached_bytes);
  EXPECT_THAT(dev_->Resolve(p), IsOk());
  EXPECT_THAT(dev_->Deallocate(p, 5000), IsOk());
  const uint64_t cached = dev_->memory_stats().cached_bytes;
  EXPECT_THAT(dev_->Deallocate(p, 5000),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("double free")));
  EXPECT_EQ(dev_->memory_stats().cached_bytes, cached);  // not cached twice
}

// METAL_PJRT_DEBUG_FREE_QUARANTINE: a freed buffer is not reused right away,
// and a pointer into it is reported with where it was freed.
TEST_F(MetalRuntimeTest, FreeQuarantine) {
  setenv("METAL_PJRT_DEBUG_FREE_QUARANTINE", "1", 1);
  absl::StatusOr<std::unique_ptr<Device>> d = Device::Create(0);
  unsetenv("METAL_PJRT_DEBUG_FREE_QUARANTINE");
  ASSERT_THAT(d, IsOk());
  Device& dev = **d;
  absl::StatusOr<Allocation> a = dev.Allocate(4096);
  ASSERT_THAT(a, IsOk());
  ASSERT_THAT(dev.Deallocate(a->ptr, 4096), IsOk());
  absl::StatusOr<Allocation> b = dev.Allocate(4096);
  ASSERT_THAT(b, IsOk());
  EXPECT_NE(b->ptr, a->ptr);
  EXPECT_FALSE(dev.Resolve(a->ptr).ok());
  EXPECT_THAT(dev.Deallocate(static_cast<char*>(a->ptr) + 8),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("inside a quarantined buffer")));
  // After kQuarantineFrees more frees and kQuarantineTime it is reusable.
  for (int i = 0; i < Device::kQuarantineFrees; ++i) {
    absl::StatusOr<Allocation> c = dev.Allocate(64);
    ASSERT_THAT(c, IsOk());
    ASSERT_THAT(dev.Deallocate(c->ptr, 64), IsOk());
  }
  std::this_thread::sleep_for(Device::kQuarantineTime +
                              std::chrono::milliseconds(20));
  absl::StatusOr<Allocation> again = dev.Allocate(4096);
  ASSERT_THAT(again, IsOk());
  EXPECT_EQ(again->ptr, a->ptr);
  EXPECT_THAT(dev.Deallocate(again->ptr, 4096), IsOk());
  EXPECT_THAT(dev.Deallocate(b->ptr, 4096), IsOk());
  // An idle trim releases quarantined buffers once kQuarantineTime passed.
  std::this_thread::sleep_for(Device::kQuarantineTime +
                              std::chrono::milliseconds(20));
  dev.TrimCache(std::chrono::seconds(0));
  EXPECT_EQ(dev.memory_stats().cached_bytes, 0u);
}

// Host memory beyond maxBufferLength stays host-only, and says so.
TEST_F(MetalRuntimeTest, HugeHostMemoryIsHostOnly) {
  const uint64_t size = dev_->info().max_buffer_length + 1;
  absl::StatusOr<void*> p = dev_->AllocateHost(size);  // pages untouched
  ASSERT_THAT(p, IsOk());
  EXPECT_TRUE(dev_->IsHostMemory(static_cast<char*>(*p) + 12345));
  EXPECT_THAT(dev_->Resolve(*p),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("too large for one Metal buffer")));
  EXPECT_THAT(dev_->DeallocateHost(*p), IsOk());
  EXPECT_FALSE(dev_->IsHostMemory(*p));
}

// A buffer the host writes right away (module constants, FFT tables) never
// reuses a cached buffer that queued GPU work may still read; GPU use may.
TEST_F(MetalRuntimeTest, HostWriteAllocationSkipsBuffersInUse) {
  const uint32_t n = 1 << 20;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  for (int i = 0; i < 300; ++i) {
    ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  }
  ASSERT_THAT(s->Flush(), IsOk());
  // XLA frees inputs as soon as their work is enqueued.
  ASSERT_THAT(dev_->Deallocate(x), IsOk());
  absl::StatusOr<Allocation> host = dev_->Allocate(n * 4, Device::Use::kHostWrite);
  ASSERT_THAT(host, IsOk());
  EXPECT_NE(host->ptr, x);
  std::memset(host->ptr, 0, n * 4);  // must not reach the queued kernels
  absl::StatusOr<Allocation> gpu = dev_->Allocate(n * 4);
  ASSERT_THAT(gpu, IsOk());
  EXPECT_EQ(gpu->ptr, x);  // stream-ordered reuse is fine
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[0], 300.0f);
  EXPECT_EQ(static_cast<float*>(y)[n - 1], 300.0f);
  // Once that work has ended, a host write may take the cached buffer.
  ASSERT_THAT(dev_->Deallocate(gpu->ptr), IsOk());
  absl::StatusOr<Allocation> again =
      dev_->Allocate(n * 4, Device::Use::kHostWrite);
  ASSERT_THAT(again, IsOk());
  EXPECT_EQ(again->ptr, x);
  EXPECT_THAT(dev_->Deallocate(again->ptr), IsOk());
  EXPECT_THAT(dev_->Deallocate(host->ptr), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// WaitForStream covers the other stream's host tasks, not only its committed
// GPU work: XLA orders a host-to-device copy (a host task) before compute
// on another stream exactly this way.
TEST_F(MetalRuntimeTest, WaitForStreamCoversHostTasks) {
  const uint32_t n = 1 << 16;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  ASSERT_THAT(a->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(a->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(a->Synchronize(), IsOk());
  const uint64_t holds = dev_->encode_host_waits();
  for (int round = 0; round < 2; ++round) {
    // Round 1 also has GPU work committed on `a` before the task.
    if (round == 1) ASSERT_THAT(Axpy(a.get(), x, y, n, 0.0f, n / 256), IsOk());
    ASSERT_THAT(a->HostCallback([y, n, round]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      for (uint32_t i = 0; i < n; ++i) {
        static_cast<float*>(y)[i] = 5.0f + round;
      }
      return absl::OkStatus();
    }), IsOk());
    ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
    ASSERT_THAT(Axpy(b.get(), x, y, n, 1.0f, n / 256), IsOk());  // y += 1
    ASSERT_THAT(b->Synchronize(), IsOk());
    EXPECT_EQ(static_cast<float*>(y)[0], 6.0f + round) << round;
    EXPECT_EQ(static_cast<float*>(y)[n - 1], 6.0f + round) << round;
  }
  EXPECT_GT(dev_->encode_host_waits(), holds);
  ASSERT_THAT(a->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// WaitForStream is transitive: B waiting for A also waits for what A was
// told to wait for, even when A has launched nothing since (CUDA orders B
// after C here too). C's producer is a host task: GPU-only producers are
// also ordered by Metal's hazard tracking of the shared buffer, which would
// hide the bug.
TEST_F(MetalRuntimeTest, WaitForStreamIncludesTheOtherStreamsWaits) {
  const uint32_t n = 1 << 16;
  void* y = Alloc(n * 4);
  void* z = Alloc(n * 4);
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  std::unique_ptr<Stream> c = NewStream();
  ASSERT_THAT(c->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(c->Synchronize(), IsOk());
  const uint64_t holds = dev_->encode_host_waits();
  for (int round = 1; round <= 2; ++round) {
    // Round 2: `a` also has committed work of its own.
    if (round == 2) ASSERT_THAT(a->Memset32(z, 0, n * 4), IsOk());
    ASSERT_THAT(c->HostCallback([y, n, round]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      for (uint32_t i = 0; i < n; ++i) {
        static_cast<float*>(y)[i] = 100.0f * round;
      }
      return absl::OkStatus();
    }), IsOk());
    ASSERT_THAT(a->WaitForStream(c.get()), IsOk());
    ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
    ASSERT_THAT(b->MemcpyDeviceToDevice(z, y, n * 4), IsOk());
    ASSERT_THAT(b->Synchronize(), IsOk());
    EXPECT_EQ(static_cast<float*>(z)[0], 100.0f * round) << round;
    EXPECT_EQ(static_cast<float*>(z)[n - 1], 100.0f * round) << round;
    ASSERT_THAT(a->Synchronize(), IsOk());
  }
  EXPECT_GT(dev_->encode_host_waits(), holds);
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
  EXPECT_THAT(dev_->Deallocate(z), IsOk());
}

// A host task that waits for its own stream gets an error, not a deadlock.
TEST_F(MetalRuntimeTest, HostTaskWaitingOnItsOwnStreamFails) {
  const uint32_t n = 1024;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  absl::StatusOr<std::unique_ptr<Event>> ev = dev_->CreateEvent();
  ASSERT_THAT(ev, IsOk());
  Event* e = ev->get();
  std::atomic<bool> go{false};
  absl::Status sync_status, commit_status, event_status;
  ASSERT_THAT(a->HostCallback([&]() {
    sync_status = a->Synchronize();
    for (int i = 0; i < 2000 && !go.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // `b` waits for this task; encoding its work from here would wait for
    // the task itself, as would waiting for the event recorded after it.
    commit_status = Axpy(b.get(), x, y, n, 1.0f, n / 256);
    event_status = e->WaitOnHost();
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
  ASSERT_THAT(a->RecordEvent(e), IsOk());  // includes the task
  go = true;
  ASSERT_THAT(a->Synchronize(), IsOk());
  EXPECT_THAT(sync_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(commit_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(event_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(e->WaitOnHost(), IsOk());
  // The work refused inside the task is committed later from outside it.
  EXPECT_THAT(b->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A host task (or its on_error) calling into its own stream is refused
// before it takes the stream lock: here the main thread's Axpy holds that
// lock while it waits for the task, so taking it would hang.
TEST_F(MetalRuntimeTest, HostTaskCallingIntoItsOwnStreamIsRefused) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  std::atomic<bool> flushing{false};
  absl::Status memset_status, callback_status, on_error_status;
  ASSERT_THAT(s->HostCallback([&]() {
    for (int i = 0; i < 2000 && !flushing.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    memset_status = s->Memset32(x, 0, n * 4);
    callback_status = s->HostCallback([]() { return absl::OkStatus(); });
    return absl::DataLossError("handled by on_error");
  }, [&](absl::Status) { on_error_status = s->Memset32(x, 0, n * 4); }),
              IsOk());
  flushing = true;
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(memset_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(callback_status,
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(on_error_status,
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(dev_->error(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// Work a host task (or a device-to-host copy) waits for is committed when
// the task is enqueued: it runs without a sync, also when the open buffer
// holds fewer ops than an early commit needs.
TEST_F(MetalRuntimeTest, HostTasksDoNotWaitForTheNextCommit) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());  // open buffer
  std::atomic<float> seen{-1.0f};
  ASSERT_THAT(s->HostCallback([&]() {
    seen = static_cast<float*>(y)[n - 1];
    return absl::OkStatus();
  }), IsOk());
  std::vector<float> out(n, -1.0f);
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(s->MemcpyDeviceToHost(out.data(), y, n * 4), IsOk());
  // The copy writes `out` on the worker; this flag (set by the next task)
  // orders that write before the read below.
  std::atomic<bool> copied{false};
  ASSERT_THAT(s->HostCallback([&]() {
    copied = true;
    return absl::OkStatus();
  }), IsOk());
  for (int i = 0; i < 2000 && (seen.load() < 0 || !copied.load()); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(seen.load(), 1.0f);
  ASSERT_TRUE(copied.load());
  EXPECT_EQ(out[n - 1], 2.0f);
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A stream waiting on the host for another stream's host task does not hold
// the queue: other streams keep encoding and completing meanwhile.
TEST_F(MetalRuntimeTest, HostWaitsDoNotHoldTheQueue) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  std::unique_ptr<Stream> c = NewStream();
  ASSERT_THAT(c->Memset32(x, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(c->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(c->Synchronize(), IsOk());
  ASSERT_THAT(a->HostCallback([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
  std::thread blocked([&]() {  // waits ~500 ms in the encode
    EXPECT_THAT(Axpy(b.get(), x, x, n, 1.0f, n / 256), IsOk());
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_THAT(Axpy(c.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(c->Synchronize(), IsOk());
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  EXPECT_LT(ms, 250);
  EXPECT_EQ(static_cast<float*>(y)[0], 1.0f);
  blocked.join();
  ASSERT_THAT(b->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(x)[0], 2.0f);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// The budget refuses allocations beyond live + cached, evicting cached
// buffers first; freed buffers are recycled by size, released after
// Device::kCacheIdleRelease unused, and not cached under memory pressure.
TEST_F(MetalRuntimeTest, MemoryBudgetAndCache) {
  const uint64_t mb = 1 << 20;
  const double fraction = 64.0 * mb / dev_->memory_budget();
  setenv("METAL_PJRT_MEMORY_FRACTION", std::to_string(fraction).c_str(), 1);
  absl::StatusOr<std::unique_ptr<Device>> made = Device::Create(0);
  unsetenv("METAL_PJRT_MEMORY_FRACTION");
  ASSERT_THAT(made, IsOk());
  Device& d = **made;
  EXPECT_NEAR(static_cast<double>(d.memory_budget()), 64.0 * mb, mb);

  absl::StatusOr<Allocation> a = d.Allocate(40 * mb);
  ASSERT_THAT(a, IsOk());
  EXPECT_THAT(d.Allocate(40 * mb),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("memory budget")));
  const uint64_t released = d.memory_stats().released;
  ASSERT_THAT(d.Deallocate(a->ptr), IsOk());
  EXPECT_EQ(d.memory_stats().cached_bytes, 40 * mb);
  // A hit: the same buffer, and no release.
  absl::StatusOr<Allocation> b = d.Allocate(40 * mb - 100);
  ASSERT_THAT(b, IsOk());
  EXPECT_EQ(b->ptr, a->ptr);
  EXPECT_EQ(d.memory_stats().cache_hits, 1u);
  ASSERT_THAT(d.Deallocate(b->ptr), IsOk());
  EXPECT_EQ(d.memory_stats().released, released);
  // A miss that only fits once the cached 40 MB is evicted.
  absl::StatusOr<Allocation> c = d.Allocate(48 * mb);
  ASSERT_THAT(c, IsOk());
  Device::MemoryStats m = d.memory_stats();
  EXPECT_EQ(m.live_bytes, 48 * mb);
  EXPECT_EQ(m.cached_bytes, 0u);
  EXPECT_GT(d.memory_stats().released, released);
  ASSERT_THAT(d.Deallocate(c->ptr), IsOk());

  // The idle timer releases what nobody reused.
  EXPECT_EQ(d.memory_stats().cached_bytes, 48 * mb);
  std::this_thread::sleep_for(Device::kCacheIdleRelease +
                              std::chrono::milliseconds(1500));
  EXPECT_EQ(d.memory_stats().cached_bytes, 0u);

  // Under pressure: the cache is dropped and frees release immediately.
  absl::StatusOr<Allocation> e = d.Allocate(mb);
  absl::StatusOr<Allocation> f = d.Allocate(mb);
  ASSERT_THAT(e, IsOk());
  ASSERT_THAT(f, IsOk());
  ASSERT_THAT(d.Deallocate(e->ptr), IsOk());
  EXPECT_EQ(d.memory_stats().cached_bytes, mb);
  d.OnMemoryPressure(1);
  EXPECT_EQ(d.memory_stats().cached_bytes, 0u);
  ASSERT_THAT(d.Deallocate(f->ptr), IsOk());
  m = d.memory_stats();
  EXPECT_EQ(m.cached_bytes, 0u);
  EXPECT_EQ(m.live_bytes, 0u);
  EXPECT_EQ(m.pressure, 1);
  d.OnMemoryPressure(0);
  absl::StatusOr<Allocation> g = d.Allocate(mb);
  ASSERT_THAT(g, IsOk());
  ASSERT_THAT(d.Deallocate(g->ptr), IsOk());
  EXPECT_EQ(d.memory_stats().cached_bytes, mb);
}

// A freed buffer that a host task still uses (host tasks hold raw
// pointers) stays cached through the pressure handler, a trim and the idle
// timer, and is released once the task has run.
TEST_F(MetalRuntimeTest, CacheReleaseWaitsForHostTasks) {
  const uint64_t size = 4 << 20;
  void* x = Alloc(size);
  std::unique_ptr<Stream> s = NewStream();
  std::atomic<bool> done{false};
  ASSERT_THAT(s->HostCallback([x, size, &done]() {
    std::this_thread::sleep_for(Device::kCacheIdleRelease +
                                std::chrono::milliseconds(2000));
    std::memset(x, 7, size);  // a use after free if x had been released
    done = true;
    return absl::OkStatus();
  }),
              IsOk());
  const uint64_t released = dev_->memory_stats().released;
  ASSERT_THAT(dev_->Deallocate(x), IsOk());
  dev_->OnMemoryPressure(1);
  dev_->TrimCache(std::chrono::seconds(0));
  std::this_thread::sleep_for(Device::kCacheIdleRelease +
                              std::chrono::milliseconds(1200));
  EXPECT_FALSE(done.load());
  EXPECT_EQ(dev_->memory_stats().released, released);
  EXPECT_EQ(dev_->memory_stats().cached_bytes, size);
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_TRUE(done.load());
  dev_->TrimCache(std::chrono::seconds(0));
  EXPECT_EQ(dev_->memory_stats().cached_bytes, 0u);
  EXPECT_GT(dev_->memory_stats().released, released);
  dev_->OnMemoryPressure(0);
}

// Host transfers run on the calling thread when the stream is idle, and in
// stream order behind pending work.
TEST_F(MetalRuntimeTest, HostTransfersInlineWhenIdle) {
  const uint32_t n = 1 << 16;
  float* d = static_cast<float*>(Alloc(n * 4));
  std::unique_ptr<Stream> s = NewStream();
  std::vector<float> src(n, 1.0f);
  ASSERT_THAT(s->MemcpyHostToDevice(d, src.data(), n * 4), IsOk());
  EXPECT_EQ(d[n - 1], 1.0f);  // idle: already copied, no sync needed
  // Busy: a host task ahead of the copy.
  ASSERT_THAT(s->HostCallback([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return absl::OkStatus();
  }),
              IsOk());
  std::vector<float> two(n, 2.0f);
  ASSERT_THAT(s->MemcpyHostToDevice(d, two.data(), n * 4), IsOk());
  EXPECT_EQ(d[0], 1.0f);  // not yet
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(d[0], 2.0f);
  EXPECT_EQ(d[n - 1], 2.0f);
  std::vector<float> back(n);
  ASSERT_THAT(s->MemcpyDeviceToHost(back.data(), d, n * 4), IsOk());
  EXPECT_EQ(back[n - 1], 2.0f);  // idle again: inline
  EXPECT_THAT(dev_->Deallocate(d), IsOk());
}

// A host transfer whose device side is not a live allocation (freed, or
// past the end of one) is refused, not written through.
TEST_F(MetalRuntimeTest, HostTransfersRefuseUnregisteredPointers) {
  const uint32_t n = 1 << 12;
  float* d = static_cast<float*>(Alloc(n * 4));
  std::unique_ptr<Stream> s = NewStream();
  std::vector<float> host(2 * n, 1.0f);
  EXPECT_THAT(s->MemcpyHostToDevice(d, host.data(), 2 * n * 4),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("past the end of a live allocation")));
  EXPECT_THAT(s->MemcpyDeviceToHost(host.data(), d + 1, n * 4),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("past the end of a live allocation")));
  ASSERT_THAT(s->MemcpyHostToDevice(d, host.data(), n * 4), IsOk());
  ASSERT_THAT(dev_->Deallocate(d), IsOk());  // cached, no longer live
  EXPECT_THAT(s->MemcpyHostToDevice(d, host.data(), n * 4),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not inside a live allocation")));
  EXPECT_THAT(s->MemcpyDeviceToHost(host.data(), d, n * 4),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not inside a live allocation")));
  ASSERT_THAT(s->Synchronize(), IsOk());  // a refused copy is no GPU error
}

// A failed command buffer's error is sticky for the device: its events,
// work that depends on it (also on other streams), unrelated streams, new
// launches and host tasks all get it, and nothing recovers.
TEST_F(MetalRuntimeTest, GpuErrorIsStickyForTheDevice) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  std::unique_ptr<Stream> s2 = NewStream();
  std::unique_ptr<Stream> s3 = NewStream();
  auto new_event = [&]() {
    absl::StatusOr<std::unique_ptr<Event>> e = dev_->CreateEvent();
    EXPECT_THAT(e, IsOk());
    return *std::move(e);
  };
  std::unique_ptr<Event> e1 = new_event(), e2 = new_event();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());

  s->FailNextCommandBufferForTesting(absl::InternalError("injected fault"));
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(s->RecordEvent(e1.get()), IsOk());  // the failing buffer
  // Work on another stream waiting for the event: refused if the error is
  // already recorded, else it waits for the force-signal; either way the
  // injected error comes back.
  absl::Status dependent = s2->WaitForEvent(e1.get());
  if (dependent.ok()) dependent = Axpy(s2.get(), x, y, n, 1.0f, n / 256);
  if (dependent.ok()) dependent = s2->RecordEvent(e2.get());
  if (dependent.ok()) dependent = e2->WaitOnHost();
  EXPECT_THAT(dependent, StatusIs(absl::StatusCode::kInternal,
                                  HasSubstr("injected fault")));
  EXPECT_THAT(e1->WaitOnHost(), StatusIs(absl::StatusCode::kInternal,
                                         HasSubstr("injected fault")));
  EXPECT_FALSE(e1->Poll().ok());
  EXPECT_THAT(dev_->error(),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("restart the Python process"),
                             HasSubstr("Earlier GPU failure: "))));
  // New launches anywhere are refused.
  EXPECT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(Axpy(s3.get(), x, y, n, 1.0f, n / 256),
              StatusIs(absl::StatusCode::kInternal));
  // Host tasks: without on_error the task still runs (XLA's callbacks free
  // memory or complete transfers); with on_error it does not, and on_error
  // gets the error.
  bool ran_anyway = false;
  ASSERT_THAT(s3->HostCallback([&]() {
    ran_anyway = true;
    return absl::OkStatus();
  }), IsOk());
  bool ran = false;
  absl::Status task_error;
  ASSERT_THAT(s3->HostCallback(
                  [&]() {
                    ran = true;
                    return absl::OkStatus();
                  },
                  [&](absl::Status st) { task_error = st; }),
              IsOk());
  EXPECT_THAT(s3->Synchronize(), StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE(ran_anyway);
  EXPECT_FALSE(ran);
  EXPECT_THAT(task_error, StatusIs(absl::StatusCode::kInternal,
                                   HasSubstr("injected fault")));
  // Sticky: every Synchronize keeps reporting it.
  for (Stream* st : {s.get(), s2.get(), s3.get(), s.get()}) {
    EXPECT_THAT(st->Synchronize(), StatusIs(absl::StatusCode::kInternal,
                                            HasSubstr("injected fault")));
  }
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// After a failure a transfer task skips its copy (XLA may already be
// freeing its buffers), and waiters on it still wait for it to end.
TEST_F(MetalRuntimeTest, TransferTaskSkipsItsCopyAfterAFailure) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  float* y = static_cast<float*>(Alloc(n * 4));
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  s->FailNextCommandBufferForTesting(absl::InternalError("injected fault"));
  ASSERT_THAT(Axpy(s.get(), x, x, n, 1.0f, n / 256), IsOk());
  // Not idle: the copy is a host task behind the failing buffer.
  std::vector<float> src(n, 5.0f);
  ASSERT_THAT(s->MemcpyHostToDevice(y, src.data(), n * 4), IsOk());
  EXPECT_THAT(s->Synchronize(), StatusIs(absl::StatusCode::kInternal,
                                         HasSubstr("injected fault")));
  EXPECT_EQ(y[0], 0.0f);
  EXPECT_EQ(y[n - 1], 0.0f);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A wait whose GPU part failed still waits for its host tasks: a task may
// still be using memory its waiter frees as soon as the wait returns (a
// device-to-host copy mid-memcpy under BlockHostUntilDone; a plain task an
// other stream's task waits for). The failing buffer is followed by ~0.5 s
// of queued GPU work, so the waits' GPU values are still unsignaled when the
// error is recorded (a failed buffer itself is force-signaled). The tasks
// have no on_error, so they run after the failure.
TEST_F(MetalRuntimeTest, WaitsAfterAFailureStillWaitForHostTasks) {
  const uint32_t n = 1 << 22;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  std::unique_ptr<Stream> s2 = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  s->FailNextCommandBufferForTesting(absl::InternalError("injected fault"));
  // ~0.4 ms each; launches are refused once the error is recorded.
  int launched = 0;
  for (; launched < 1200; ++launched) {
    if (!Axpy(s.get(), x, y, n, 1.0f, n / 256).ok()) break;
  }
  std::atomic<bool> done{false};
  std::atomic<bool> seen_done{false};
  ASSERT_THAT(s->HostCallback([&done]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    done = true;
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(s2->WaitForStream(s.get()), IsOk());
  ASSERT_THAT(s2->HostCallback([&]() {
    seen_done = done.load();
    return absl::OkStatus();
  }), IsOk());
  EXPECT_THAT(s->Synchronize(), StatusIs(absl::StatusCode::kInternal,
                                         HasSubstr("injected fault")));
  EXPECT_TRUE(done.load()) << launched << " launches";
  EXPECT_THAT(s2->Synchronize(), StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE(seen_done.load());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A GPU op whose encode fails keeps the stream's waits: a later host task on
// the stream still runs only after the GPU work the stream waited for.
TEST_F(MetalRuntimeTest, FailedEncodeKeepsTheStreamsWaits) {
  const uint32_t n = 1 << 20;
  void* x = Alloc(n * 4);
  float* y = static_cast<float*>(Alloc(n * 4));
  std::unique_ptr<Stream> a = NewStream();
  std::unique_ptr<Stream> b = NewStream();
  ASSERT_THAT(a->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(a->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(a->Synchronize(), IsOk());
  const int kLaunches = 500;  // ~50 ms
  for (int i = 0; i < kLaunches; ++i) {
    ASSERT_THAT(Axpy(a.get(), x, y, n, 1.0f, n / 256), IsOk());
  }
  ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
  EXPECT_THAT(b->EncodeExternal([](void*) {
    return absl::InvalidArgumentError("encode failed");
  }, /*flops=*/0), StatusIs(absl::StatusCode::kInvalidArgument));
  std::atomic<float> seen{-1.0f};
  ASSERT_THAT(b->HostCallback([&]() {
    seen = y[n - 1];
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(b->Synchronize(), IsOk());
  EXPECT_EQ(seen.load(), static_cast<float>(kLaunches));
  EXPECT_THAT(dev_->error(), IsOk());
  ASSERT_THAT(a->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// GEMM launches are charged their flops: the buffer is committed once the
// budget is reached, even with few ops and threads (and the GPU busy). The
// kernel is a tiny axpy; only its declared cost is large.
TEST_F(MetalRuntimeTest, FlopsBudgetCommits) {
  const uint32_t n = 256;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0, n * 4), IsOk());
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  Params p{n, 1.0f};
  auto launch = [&](uint64_t flops) {
    return s->Launch(*kernel_, Dim3{1, 1, 1}, Dim3{256, 1, 1},
                     {KernelArg::Buffer(x), KernelArg::Buffer(y),
                      KernelArg::Bytes(&p, sizeof(p))},
                     0, flops);
  };
  const uint64_t committed = s->FenceForTesting().first;
  ASSERT_THAT(launch(Stream::kMaxFlopsPerCommandBuffer / 2), IsOk());
  EXPECT_EQ(s->FenceForTesting().first, committed);  // still open
  ASSERT_THAT(launch(Stream::kMaxFlopsPerCommandBuffer / 2), IsOk());
  EXPECT_GT(s->FenceForTesting().first, committed);  // budget reached
  // An op that would take the open buffer over the budget starts a new
  // one: the first is committed before it is encoded, and it stays open.
  const uint64_t before = s->FenceForTesting().first;
  ASSERT_THAT(launch(Stream::kMaxFlopsPerCommandBuffer / 4 * 3), IsOk());
  EXPECT_EQ(s->FenceForTesting().first, before);
  ASSERT_THAT(launch(Stream::kMaxFlopsPerCommandBuffer / 2), IsOk());
  EXPECT_EQ(s->FenceForTesting().first, before + 1);
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// Work encoded before a failure elsewhere is never committed: the open
// command buffer is dropped (its timeline value force-signaled), and no
// more work goes into it.
TEST_F(MetalRuntimeTest, NothingIsCommittedAfterTheStickyError) {
  const uint32_t n = 1 << 12;
  float* x = static_cast<float*>(Alloc(n * 4));
  float* y = static_cast<float*>(Alloc(n * 4));
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  // One launch stays in the open command buffer (below the commit caps).
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  // Another buffer failed meanwhile (as its completion handler reports it).
  dev_->SetError(absl::InternalError("injected fault"));

  // The open buffer takes no more work, and ending it commits nothing.
  EXPECT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(s->Flush(), IsOk());
  EXPECT_THAT(s->Synchronize(), StatusIs(absl::StatusCode::kInternal,
                                         HasSubstr("injected fault")));
  const auto [committed, signaled] = s->FenceForTesting();
  EXPECT_GE(signaled, committed);  // waiters on the dropped buffer wake up
  EXPECT_EQ(y[0], 0.0f);  // never ran on the GPU
  EXPECT_EQ(y[n - 1], 0.0f);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// A host task's own failure goes to its on_error when it has one; without
// one it becomes the device's sticky error.
TEST_F(MetalRuntimeTest, HostTaskErrors) {
  const uint32_t n = 1 << 12;
  void* x = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  std::unique_ptr<Stream> s2 = NewStream();
  absl::Status seen;
  ASSERT_THAT(s->HostCallback([]() { return absl::DataLossError("handled"); },
                              [&](absl::Status st) { seen = st; }),
              IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(seen, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(dev_->error(), IsOk());
  ASSERT_THAT(s->HostCallback([]() { return absl::DataLossError("task"); }),
              IsOk());
  EXPECT_THAT(s->Synchronize(), StatusIs(absl::StatusCode::kDataLoss,
                                         HasSubstr("task")));
  EXPECT_THAT(s2->Synchronize(), StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(s2->Memset32(x, 0, n * 4),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
}

TEST_F(MetalRuntimeTest, ErrorCodes) {
  EXPECT_THAT(Device::Create(-1),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("-1")));
  EXPECT_THAT(dev_->Allocate(dev_->info().max_buffer_length + 1),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("maxBufferLength")));
  int host = 0;
  EXPECT_THAT(dev_->Deallocate(&host),
              StatusIs(absl::StatusCode::kInternal));  // a plugin bug
  EXPECT_THAT(dev_->Resolve(&host),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(dev_->GetKernel("kernel void broken(", "broken"),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("domain=")));
  EXPECT_THAT(dev_->GetKernel(kMsl, "no_such_function"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("no_such_function")));

  std::unique_ptr<Stream> s = NewStream();
  void* x = Alloc(1024);
  EXPECT_THAT(Axpy(s.get(), x, &host, 1, 1.0f, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("argument 1")));
  std::vector<KernelArg> too_many(Stream::kMaxBufferArgs + 1,
                                  KernelArg::Buffer(x));
  EXPECT_THAT(s->Launch(*kernel_, Dim3{}, Dim3{}, too_many),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(s->Memset32(x, 0x12345678u, 6),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s->Memset8(x, 0, 4096),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // None of the rejected calls poisoned the stream.
  EXPECT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
}

// A kernel with more buffer arguments than Metal's argument table: the
// arguments travel as GPU addresses in an argument buffer.
TEST_F(MetalRuntimeTest, ArgumentBufferLaunch) {
  constexpr int kInputs = 40;
  constexpr uint32_t kN = 1000;
  std::string msl =
      "#include <metal_stdlib>\nusing namespace metal;\n";
  msl += std::string(kArgumentBufferMarker) +
         "\nkernel void sum_many(constant ulong* xla_args [[buffer(0)]],\n"
         "    uint i [[thread_position_in_grid]]) {\n"
         "  if (i >= " + std::to_string(kN) + ") return;\n"
         "  float acc = 0;\n"
         "  for (int j = 0; j < " + std::to_string(kInputs) + "; ++j) {\n"
         "    acc += ((device const float*)xla_args[j])[i];\n"
         "  }\n"
         "  ((device float*)xla_args[" + std::to_string(kInputs) +
         "])[i] = acc;\n}\n";
  EXPECT_TRUE(UsesArgumentBuffer(msl, "sum_many"));
  EXPECT_FALSE(UsesArgumentBuffer(kMsl, "axpy"));
  absl::StatusOr<const Kernel*> k = dev_->GetKernel(msl, "sum_many");
  ASSERT_THAT(k, IsOk());
  EXPECT_TRUE((*k)->uses_argument_buffer());

  // Half the inputs are separate allocations, the other half interior
  // pointers into one shared allocation (duplicate MTLBuffers).
  std::vector<void*> allocs;
  auto* shared = static_cast<float*>(Alloc(kInputs / 2 * kN * sizeof(float)));
  allocs.push_back(shared);
  std::vector<KernelArg> args;
  for (int j = 0; j < kInputs; ++j) {
    float* p;
    if (j % 2 == 0) {
      p = static_cast<float*>(Alloc(kN * sizeof(float)));
      allocs.push_back(p);
    } else {
      p = shared + (j / 2) * kN;
    }
    for (uint32_t i = 0; i < kN; ++i) p[i] = static_cast<float>(j + 1);
    args.push_back(KernelArg::Buffer(p));
  }
  auto* out = static_cast<float*>(Alloc(kN * sizeof(float)));
  allocs.push_back(out);
  args.push_back(KernelArg::Buffer(out));

  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Launch(**k, Dim3{(kN + 255) / 256, 1, 1}, Dim3{256, 1, 1},
                        args),
              IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  const float expected = kInputs * (kInputs + 1) / 2.0f;
  for (uint32_t i = 0; i < kN; ++i) ASSERT_EQ(out[i], expected) << i;
  for (void* p : allocs) EXPECT_THAT(dev_->Deallocate(p), IsOk());
}

// A declared [[max_total_threads_per_threadgroup(N)]] caps the pipeline
// limit; a larger threadgroup is refused before anything is encoded (release
// Metal would run it as undefined behaviour).
TEST_F(MetalRuntimeTest, DeclaredMaxThreadsRefusesLargerThreadgroups) {
  const std::string msl =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "[[max_total_threads_per_threadgroup(64)]]\n"
      "kernel void capped(device uint* y [[buffer(0)]],\n"
      "    uint i [[thread_position_in_grid]]) { y[i] = i; }\n";
  EXPECT_EQ(DeclaredMaxThreadsPerThreadgroup(msl, "capped"), 64u);
  EXPECT_EQ(DeclaredMaxThreadsPerThreadgroup(msl, "other"), 0u);
  EXPECT_EQ(DeclaredMaxThreadsPerThreadgroup(kMsl, "axpy"), 0u);
  EXPECT_EQ(DeclaredMaxThreadsPerThreadgroup(
                std::string("[[max_total_threads_per_threadgroup(32)]]\n") +
                    kArgumentBufferMarker + "\nkernel void k(",
                "k"),
            32u);
  absl::StatusOr<const Kernel*> k = dev_->GetKernel(msl, "capped");
  ASSERT_THAT(k, IsOk());
  RecordProperty("pso_max_threads_with_attribute_64",
                 static_cast<int>((*k)->max_total_threads_per_threadgroup()));
  // Even if the pipeline reported more, the declared value is the limit.
  EXPECT_LE((*k)->max_total_threads_per_threadgroup(), 64u);

  std::unique_ptr<Stream> s = NewStream();
  auto* y = static_cast<uint32_t*>(Alloc(256 * sizeof(uint32_t)));
  for (int i = 0; i < 256; ++i) y[i] = 12345;
  EXPECT_THAT(s->Launch(**k, Dim3{2, 1, 1}, Dim3{128, 1, 1},
                        {KernelArg::Buffer(y)}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("128 threads per threadgroup")));
  EXPECT_THAT(s->Synchronize(), IsOk());
  for (int i = 0; i < 256; ++i) ASSERT_EQ(y[i], 12345u) << "refused launch ran";
  // LaunchKernel refuses too (it used to shrink the threadgroup to 64,
  // which ran only 2 x 64 of the 2 x 128 threads asked for).
  struct NoParams {
    uint32_t unused;
  };
  EXPECT_THAT(LaunchKernel(s.get(), **k, {y}, NoParams{0}, Dim3{2, 1, 1},
                           Dim3{128, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(s->Synchronize(), IsOk());
  for (int i = 0; i < 256; ++i) ASSERT_EQ(y[i], 12345u) << "refused launch ran";
  ASSERT_THAT(s->Launch(**k, Dim3{4, 1, 1}, Dim3{64, 1, 1},
                        {KernelArg::Buffer(y)}),
              IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  for (int i = 0; i < 256; ++i) ASSERT_EQ(y[i], static_cast<uint32_t>(i));
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

}  // namespace
}  // namespace rt
}  // namespace metal_pjrt

namespace metal_pjrt::rt {
namespace {
using ::absl_testing::StatusIs;

TEST_F(MetalRuntimeTest, MemoryBudgetAndCriticalPressure) {
  // Half of RAM, capped by the working set: usable, never the whole machine.
  EXPECT_GE(dev_->memory_budget(), 256ull << 20);
  EXPECT_LE(dev_->memory_budget(), dev_->info().recommended_working_set);
  EXPECT_LE(dev_->memory_budget(), PhysicalMemoryBytes() / 2);
  // Past the budget: refused cleanly, not attempted.
  EXPECT_THAT(dev_->Allocate(dev_->memory_budget() + (1 << 20)),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(dev_->allocated_bytes(), 0);
  // At (faked) critical system pressure: 1 MB or more is refused, less is
  // not.
  SetMemoryPressureForTesting(2);
  const absl::StatusOr<Allocation> big = dev_->Allocate(16 << 20);
  const absl::StatusOr<Allocation> small = dev_->Allocate(4096);
  SetMemoryPressureForTesting(0);
  EXPECT_THAT(big, StatusIs(absl::StatusCode::kResourceExhausted,
                            HasSubstr("critical memory pressure")));
  ASSERT_THAT(small, IsOk());
  EXPECT_THAT(dev_->Deallocate(small->ptr), IsOk());
}

}  // namespace
}  // namespace metal_pjrt::rt
