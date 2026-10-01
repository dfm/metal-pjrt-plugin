// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Exercises the Metal platform through the StreamExecutor interfaces XLA
// uses: platform lookup, allocation, MSL kernel loading via the in-memory
// binary spec, launch with packed device-pointer args, memcpy and events.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "metal_pjrt/stream_executor/metal_executor.h"
#include "metal_pjrt/stream_executor/metal_platform.h"
#include "metal_pjrt/stream_executor/metal_platform_id.h"
#include "metal_pjrt/stream_executor/metal_stream.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor {
namespace metal {
namespace {

constexpr char kAxpy[] = R"(
#include <metal_stdlib>
using namespace metal;
kernel void axpy(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 uint3 tid [[thread_position_in_threadgroup]],
                 uint3 bid [[threadgroup_position_in_grid]],
                 uint3 bdim [[threads_per_threadgroup]]) {
  uint i = bid.x * bdim.x + tid.x;
  y[i] = 2.0f * x[i] + y[i];
}
)";

TEST(MetalExecutorTest, PlatformIsRegistered) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  EXPECT_EQ(platform->id(), kMetalPlatformId);
  EXPECT_GE(platform->VisibleDeviceCount(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto desc, platform->DescriptionForDevice(0));
  EXPECT_EQ(desc->device_vendor(), "Apple");
  EXPECT_EQ(desc->threads_per_warp(), 32);
  EXPECT_TRUE(desc->gpu_compute_capability().IsOneAPI());
}

TEST(MetalExecutorTest, LoadLaunchAndCopy) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  const int n = 4096;
  DeviceAddressBase x = executor->Allocate(n * sizeof(float));
  DeviceAddressBase y = executor->Allocate(n * sizeof(float));
  ASSERT_FALSE(x.is_null());
  ASSERT_FALSE(y.is_null());
  std::vector<float> hx(n, 1.5f), hy(n, 1.0f), out(n, 0.0f);
  TF_ASSERT_OK(stream->Memcpy(&x, hx.data(), n * sizeof(float)));
  TF_ASSERT_OK(stream->Memcpy(&y, hy.data(), n * sizeof(float)));

  std::vector<uint8_t> msl(kAxpy, kAxpy + sizeof(kAxpy) - 1);
  KernelLoaderSpec spec = KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
      std::move(msl), "axpy", /*arity=*/2);
  TF_ASSERT_OK_AND_ASSIGN(auto kernel, executor->LoadKernel(spec));

  std::vector<DeviceAddressBase> args = {x, y};
  auto packed = PackKernelArgs(args, /*shared_mem_bytes=*/0);
  TF_ASSERT_OK(kernel->Launch(ThreadDim(256), BlockDim(n / 256), stream.get(),
                              *packed));
  TF_ASSERT_OK(stream->Memcpy(out.data(), y, n * sizeof(float)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  for (int i = 0; i < n; ++i) ASSERT_FLOAT_EQ(out[i], 4.0f) << i;

  // Events across streams.
  TF_ASSERT_OK_AND_ASSIGN(auto stream2, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto event, executor->CreateEvent());
  TF_ASSERT_OK(kernel->Launch(ThreadDim(256), BlockDim(n / 256), stream.get(),
                              *packed));  // y = 7
  TF_ASSERT_OK(stream->RecordEvent(event.get()));
  TF_ASSERT_OK(stream2->WaitFor(event.get()));
  TF_ASSERT_OK(stream2->Memcpy(out.data(), y, n * sizeof(float)));
  TF_ASSERT_OK(stream2->BlockHostUntilDone());
  EXPECT_EQ(event->PollForStatus(), Event::Status::kComplete);
  for (int i = 0; i < n; i += 97) ASSERT_FLOAT_EQ(out[i], 7.0f) << i;

  executor->Deallocate(&x);
  executor->Deallocate(&y);
}

TEST(MetalExecutorTest, ConstantsModule) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  // Built by hand to match runtime/constants_container.h.
  std::vector<uint8_t> blob;
  auto put = [&](const void* p, size_t k) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    blob.insert(blob.end(), b, b + k);
  };
  put("MTLCONST", 8);
  uint32_t v = 1, count = 1, name_len = 3, reserved = 0;
  uint64_t data_len = 8;
  put(&v, 4); put(&count, 4);
  put(&name_len, 4); put(&reserved, 4); put(&data_len, 8);
  put("abc", 3); blob.resize((blob.size() + 7) & ~size_t{7});
  float data[2] = {1.0f, 2.0f};
  put(data, 8);
  MultiModuleLoaderSpec spec;
  spec.AddCudaCubinInMemory(blob);
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle handle, executor->LoadModule(spec));
  TF_ASSERT_OK_AND_ASSIGN(DeviceAddressBase sym,
                          executor->GetSymbol("abc", handle));
  EXPECT_EQ(sym.size(), 8);
  EXPECT_FLOAT_EQ(static_cast<float*>(sym.opaque())[1], 2.0f);
  EXPECT_TRUE(executor->UnloadModule(handle));
}

TEST(MetalExecutorTest, AllocationFailureReturnsNull) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  DeviceAddressBase mem = executor->Allocate(uint64_t{1} << 62, 0);
  EXPECT_TRUE(mem.is_null());
  EXPECT_EQ(executor->HostMemoryAllocate(uint64_t{1} << 62).status().code(),
            absl::StatusCode::kResourceExhausted);
}

// Host memory (XLA's host pools, transfer staging) is plain host pages the
// GPU can reach: not a device allocation, not in the budget or the device's
// buffer cache.
TEST(MetalExecutorTest, HostMemoryIsNotDeviceMemory) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  metal_pjrt::rt::Device* device =
      static_cast<MetalExecutor*>(executor)->device();
  const uint64_t live = device->memory_stats().live_bytes;
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          executor->CreateMemoryAllocator(MemorySpace::kHost));
  TF_ASSERT_OK_AND_ASSIGN(auto host, allocator->Allocate(3 << 20));
  ASSERT_NE(host->opaque(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(host->opaque()) % 4096, 0u);
  std::memset(host->opaque(), 7, 3 << 20);
  EXPECT_EQ(device->memory_stats().live_bytes, live);
  TF_ASSERT_OK_AND_ASSIGN(metal_pjrt::rt::BufferRef ref,
                          device->Resolve(static_cast<char*>(host->opaque()) +
                                          100));
  EXPECT_EQ(ref.offset, 100u);
  EXPECT_NE(ref.buffer, nullptr);
  EXPECT_FALSE(device->Deallocate(host->opaque()).ok());
  TF_ASSERT_OK_AND_ASSIGN(MemorySpace space,
                          executor->GetPointerMemorySpace(host->opaque()));
  EXPECT_EQ(space, MemorySpace::kHost);
  void* ptr = host->opaque();
  host.reset();
  EXPECT_FALSE(device->Resolve(ptr).ok());
  // The device-side spaces still allocate device memory.
  TF_ASSERT_OK_AND_ASSIGN(auto dev_allocator, executor->CreateMemoryAllocator(
                                                  MemorySpace::kCollective));
  TF_ASSERT_OK_AND_ASSIGN(auto dev, dev_allocator->Allocate(1 << 20));
  EXPECT_TRUE(device->Resolve(dev->opaque()).ok());
}

// Allocation churn as XLA produces it (the same sizes every step): after
// the first round every allocation is a cache hit on the same buffer, no
// buffer is released (the released count stays), and the accounting
// balances; a trim then releases everything.
TEST(MetalExecutorTest, AllocationChurnRecyclesBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  metal_pjrt::rt::Device* device =
      static_cast<MetalExecutor*>(executor)->device();
  using Stats = metal_pjrt::rt::Device::MemoryStats;
  device->TrimCache(std::chrono::seconds(0));
  const Stats before = device->memory_stats();
  const std::vector<uint64_t> sizes = {1,       300,           4096,
                                       16385,   1 << 20,       (1 << 20) + 5,
                                       8 << 20, (8 << 20) + 1, 3};
  constexpr int kRounds = 50;
  std::vector<void*> first;
  uint64_t released = 0;
  for (int round = 0; round < kRounds; ++round) {
    std::vector<DeviceAddressBase> mems;
    for (uint64_t size : sizes) {
      DeviceAddressBase m = executor->Allocate(size, 0);
      ASSERT_FALSE(m.is_null());
      std::memset(m.opaque(), round, size);
      mems.push_back(m);
    }
    std::vector<void*> ptrs;
    for (const DeviceAddressBase& m : mems) ptrs.push_back(m.opaque());
    if (round == 0) {
      first = ptrs;
      released = device->memory_stats().released;
    } else {
      EXPECT_EQ(ptrs, first) << "round " << round;
    }
    for (DeviceAddressBase& m : mems) executor->Deallocate(&m);
  }
  const Stats after = device->memory_stats();
  EXPECT_EQ(after.cache_misses - before.cache_misses, sizes.size());
  EXPECT_EQ(after.cache_hits - before.cache_hits,
            (kRounds - 1) * sizes.size());
  EXPECT_EQ(after.live_bytes, before.live_bytes);
  EXPECT_GE(after.cached_bytes - before.cached_bytes, 18u << 20);
  EXPECT_EQ(device->memory_stats().released, released);
  device->TrimCache(std::chrono::seconds(0));
  EXPECT_EQ(device->memory_stats().cached_bytes, 0u);
  EXPECT_GT(device->memory_stats().released, released);
}

// Host callback errors and a GPU failure. The failure is sticky for the
// device, which the platform shares across this binary's tests: keep this
// test last.
TEST(MetalExecutorTest, ErrorsAreStickyForTheDevice) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto event, executor->CreateEvent());
  DeviceAddressBase mem = executor->Allocate(4096, 0);
  ASSERT_FALSE(mem.is_null());

  // With an error_cb a callback's failure goes there and nothing else fails.
  absl::Status seen;
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      [] { return absl::DataLossError("handled"); },
      [&seen](absl::Status s) { seen = std::move(s); }));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(seen.code(), absl::StatusCode::kDataLoss);

  // A GPU failure reaches the host callbacks enqueued after it (their
  // error_cb; XLA marks buffer definition events failed this way), events
  // and BlockHostUntilDone, for good. The stream's own error state (ok())
  // is never set: XLA CHECKs it on pooled streams.
  static_cast<MetalStream*>(stream.get())
      ->rt_stream()
      ->FailNextCommandBufferForTesting(absl::InternalError("injected"));
  TF_ASSERT_OK(stream->Memset32(&mem, 0x3f800000u, 4096));
  TF_ASSERT_OK(stream->RecordEvent(event.get()));
  bool ran = false;
  seen = absl::OkStatus();
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      [&ran] {
        ran = true;
        return absl::OkStatus();
      },
      [&seen](absl::Status s) { seen = std::move(s); }));
  absl::Status s = stream->BlockHostUntilDone();
  EXPECT_EQ(s.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(s.message(), ::testing::HasSubstr("injected"));
  EXPECT_FALSE(ran);
  EXPECT_EQ(seen.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(event->PollForStatus(), Event::Status::kError);
  EXPECT_EQ(event->Synchronize().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(stream->ok());
  EXPECT_EQ(stream->BlockHostUntilDone().code(), absl::StatusCode::kInternal);
  EXPECT_FALSE(stream->Memset32(&mem, 0, 4096).ok());
  executor->Deallocate(&mem);
}

}  // namespace
}  // namespace metal
}  // namespace stream_executor
