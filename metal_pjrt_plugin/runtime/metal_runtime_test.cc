// Tests for the runtime layer on its own (no XLA). Needs a Metal device.
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/runtime/system_memory.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
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

namespace metal_pjrt {
namespace rt {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
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
    absl::StatusOr<MTL::Library*> lib = dev_->CompileLibrary(kMsl);
    ASSERT_THAT(lib, IsOk());
    lib_ = *lib;
    absl::StatusOr<std::unique_ptr<Kernel>> k = dev_->CreateKernel(lib_, "axpy");
    ASSERT_THAT(k, IsOk());
    kernel_ = *std::move(k);
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
  MTL::Library* lib_ = nullptr;
  std::unique_ptr<Kernel> kernel_;
};

TEST_F(MetalRuntimeTest, DeviceInfo) {
  EXPECT_GE(Device::VisibleDeviceCount(), 1);
  EXPECT_FALSE(dev_->info().name.empty());
  EXPECT_GT(dev_->info().max_buffer_length, 0u);
  EXPECT_EQ(kernel_->thread_execution_width(), 32u);
}

TEST_F(MetalRuntimeTest, LibraryCache) {
  absl::StatusOr<MTL::Library*> again = dev_->CompileLibrary(kMsl);
  ASSERT_THAT(again, IsOk());
  EXPECT_EQ(*again, lib_);
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
  unsetenv("METAL_PJRT_QUARANTINE_STRIKES");
  absl::StatusOr<std::unique_ptr<Device>> d1 = Device::Create(0);
  ASSERT_THAT(d1, IsOk());
  EXPECT_EQ((*d1)->state_dir(), dir);
  EXPECT_EQ((*d1)->resets_since_boot(), 0);
  absl::StatusOr<MTL::Library*> lib1 = (*d1)->CompileLibrary(kMsl);
  ASSERT_THAT(lib1, IsOk());
  absl::StatusOr<std::unique_ptr<Kernel>> k1 = (*d1)->CreateKernel(*lib1, "axpy");
  ASSERT_THAT(k1, IsOk());
  EXPECT_THAT((*k1)->key(), HasSubstr(":axpy"));

  // Two resets blaming axpy (the second with a duplicate and a null entry).
  (*d1)->RecordReset("first \"timeout\"", {(*k1)->identity()});
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
  absl::StatusOr<MTL::Library*> lib2 = (*d2)->CompileLibrary(kMsl);
  ASSERT_THAT(lib2, IsOk());
  EXPECT_THAT((*d2)->CreateKernel(*lib2, "axpy"),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("quarantined")));

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
  absl::StatusOr<MTL::Library*> lib4 = (*d4)->CompileLibrary(kMsl);
  ASSERT_THAT(lib4, IsOk());
  EXPECT_THAT((*d4)->CreateKernel(*lib4, "axpy"), IsOk());
  {
    std::ofstream out(dir + "/gpu_resets.jsonl", std::ios::app);
    out << other_build(records[1]) << "\n";
  }
  absl::StatusOr<std::unique_ptr<Device>> d5 = Device::Create(0);
  ASSERT_THAT(d5, IsOk());
  absl::StatusOr<MTL::Library*> lib5 = (*d5)->CompileLibrary(kMsl);
  ASSERT_THAT(lib5, IsOk());
  EXPECT_THAT((*d5)->CreateKernel(*lib5, "axpy"),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Disabled by threshold 0.
  setenv("METAL_PJRT_QUARANTINE_STRIKES", "0", 1);
  absl::StatusOr<std::unique_ptr<Device>> d3 = Device::Create(0);
  ASSERT_THAT(d3, IsOk());
  absl::StatusOr<MTL::Library*> lib3 = (*d3)->CompileLibrary(kMsl);
  ASSERT_THAT(lib3, IsOk());
  EXPECT_THAT((*d3)->CreateKernel(*lib3, "axpy"), IsOk());
  unsetenv("METAL_PJRT_QUARANTINE_STRIKES");
  unsetenv("METAL_PJRT_STATE_DIR");
  std::filesystem::remove_all(dir);
}

TEST_F(MetalRuntimeTest, ResetLogMigratesOnceFromJaxMetalDir) {
  // Default state directory under a private HOME: ~/.cache/openmetal starts
  // with a copy of the pre-rename ~/.cache/jax_metal reset log, once.
  char tmpl[] = "/tmp/metal_rt_home_XXXXXX";
  ASSERT_NE(mkdtemp(tmpl), nullptr);
  const std::string home = tmpl;
  const char* real_home = std::getenv("HOME");
  const std::string saved_home = real_home != nullptr ? real_home : "";
  setenv("HOME", home.c_str(), 1);
  unsetenv("METAL_PJRT_STATE_DIR");
  const std::string old_log = home + "/.cache/jax_metal/gpu_resets.jsonl";
  const std::string new_log = home + "/.cache/openmetal/gpu_resets.jsonl";
  std::filesystem::create_directories(home + "/.cache/jax_metal");
  {
    std::ofstream out(old_log);
    out << "{\"time\":" << std::time(nullptr)
        << ",\"kernels\":[{\"key\":\"k1:f\"}]}\n";
  }
  absl::StatusOr<std::unique_ptr<Device>> d1 = Device::Create(0);
  ASSERT_THAT(d1, IsOk());
  EXPECT_EQ((*d1)->state_dir(), home + "/.cache/openmetal");
  EXPECT_EQ((*d1)->resets_since_boot(), 1);
  EXPECT_TRUE(std::filesystem::exists(new_log));
  EXPECT_TRUE(std::filesystem::exists(old_log));  // never touched
  // Cleared (gpu_health.py --clear truncates): not copied again.
  std::ofstream(new_log, std::ios::trunc).close();
  absl::StatusOr<std::unique_ptr<Device>> d2 = Device::Create(0);
  ASSERT_THAT(d2, IsOk());
  EXPECT_EQ((*d2)->resets_since_boot(), 0);
  setenv("HOME", saved_home.c_str(), 1);
  std::filesystem::remove_all(home);
}

// Waits are encoded lazily: a stream that only waits (for another stream,
// or for a host task) never commits a wait-only command buffer, and
// Synchronize / host tasks honor pending waits on the host.
TEST_F(MetalRuntimeTest, DeferredWaits) {
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
  // the compute stream's work even though no command buffer was committed
  // on s2 for it.
  for (int i = 0; i < 300; ++i) ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, groups), IsOk());
  ASSERT_THAT(s2->WaitForStream(s.get()), IsOk());
  ASSERT_THAT(s2->MemcpyDeviceToHost(out.data(), y, n * 4), IsOk());
  ASSERT_THAT(s2->Synchronize(), IsOk());
  EXPECT_EQ(out[0], 600.0f);
  EXPECT_EQ(out[n - 1], 600.0f);

  // A host task followed by GPU work on the same stream: the work waits for
  // the task (wait encoded lazily in front of it).
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

// Hold rule: GPU work waiting on a slow host task is committed only after
// the task has run, so no command buffer waits on host work on the GPU.
TEST_F(MetalRuntimeTest, HoldRuleKeepsHostWaitsOffTheGpu) {
  const uint32_t n = 1 << 16;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  ASSERT_THAT(s->Memset32(x, 0x3f800000u, n * 4), IsOk());  // 1.0f
  ASSERT_THAT(s->Memset32(y, 0, n * 4), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  const uint64_t holds = dev_->host_task_holds();
  ASSERT_THAT(s->HostCallback([y]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    static_cast<float*>(y)[0] = 100.0f;
    return absl::OkStatus();
  }), IsOk());
  // Encoded behind the task; its commit waits for the task on the host.
  ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
  ASSERT_THAT(s->Flush(), IsOk());
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_EQ(static_cast<float*>(y)[0], 101.0f);
  EXPECT_EQ(static_cast<float*>(y)[n - 1], 1.0f);
  EXPECT_GT(dev_->host_task_holds(), holds);
  EXPECT_EQ(dev_->unsignaled_host_task_waits_committed(), 0u);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
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
  const uint64_t holds = dev_->host_task_holds();
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
  EXPECT_GT(dev_->host_task_holds(), holds);
  EXPECT_EQ(dev_->unsignaled_host_task_waits_committed(), 0u);
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
  const uint64_t holds = dev_->host_task_holds();
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
  EXPECT_GT(dev_->host_task_holds(), holds);
  EXPECT_EQ(dev_->unsignaled_host_task_waits_committed(), 0u);
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
    // `b` waits for this task; committing its work from here would wait for
    // the task itself.
    commit_status = Axpy(b.get(), x, y, n, 1.0f, n / 256);
    if (commit_status.ok()) commit_status = b->Flush();
    // The event recorded after this task is not published until its buffer
    // is committed, so waiting for it returns at once.
    event_status = e->WaitOnHost();
    return absl::OkStatus();
  }), IsOk());
  ASSERT_THAT(b->WaitForStream(a.get()), IsOk());
  go = true;
  ASSERT_THAT(a->RecordEvent(e), IsOk());
  ASSERT_THAT(a->Synchronize(), IsOk());
  EXPECT_THAT(sync_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(commit_status, StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(event_status, IsOk());
  EXPECT_THAT(e->WaitOnHost(), IsOk());
  // The work refused inside the task is committed later from outside it.
  EXPECT_THAT(b->Synchronize(), IsOk());
  EXPECT_EQ(dev_->unsignaled_host_task_waits_committed(), 0u);
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// Commit encodes a buffer's fence signal before its event signals, so an
// event value seen on the host implies the buffer's fence value is signaled
// (Device::CheckInFlight relies on it).
TEST_F(MetalRuntimeTest, EventSignalsFollowTheFenceSignal) {
  const uint32_t n = 1 << 16;
  void* x = Alloc(n * 4);
  void* y = Alloc(n * 4);
  std::unique_ptr<Stream> s = NewStream();
  absl::StatusOr<std::unique_ptr<Event>> e = dev_->CreateEvent();
  ASSERT_THAT(e, IsOk());
  ASSERT_THAT(s->Memset32(x, 0, n * 4), IsOk());
  for (int i = 0; i < 200; ++i) {
    ASSERT_THAT(Axpy(s.get(), x, y, n, 1.0f, n / 256), IsOk());
    ASSERT_THAT(s->RecordEvent(e->get()), IsOk());
    const uint64_t fence_value = s->FenceForTesting().first;
    while (!(*e)->IsComplete()) {
    }
    ASSERT_GE(s->FenceForTesting().second, fence_value) << i;
  }
  ASSERT_THAT(s->Synchronize(), IsOk());
  EXPECT_THAT(dev_->Deallocate(x), IsOk());
  EXPECT_THAT(dev_->Deallocate(y), IsOk());
}

// The budget refuses allocations beyond live + cached, evicting cached
// buffers first; freed buffers are recycled by size, released after
// Device::kCacheIdleRelease unused, and not cached under memory pressure.
TEST_F(MetalRuntimeTest, MemoryBudgetAndCache) {
  const uint64_t mb = 1 << 20;
  const double fraction = 64.0 * mb / dev_->memory_budget();
  setenv("JAX_OPENMETAL_MEMORY_FRACTION", std::to_string(fraction).c_str(), 1);
  absl::StatusOr<std::unique_ptr<Device>> made = Device::Create(0);
  unsetenv("JAX_OPENMETAL_MEMORY_FRACTION");
  ASSERT_THAT(made, IsOk());
  Device& d = **made;
  EXPECT_NEAR(static_cast<double>(d.memory_budget()), 64.0 * mb, mb);

  absl::StatusOr<Allocation> a = d.Allocate(40 * mb);
  ASSERT_THAT(a, IsOk());
  EXPECT_THAT(d.Allocate(40 * mb),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("memory budget")));
  const uint64_t gen = d.allocation_generation();
  ASSERT_THAT(d.Deallocate(a->ptr), IsOk());
  EXPECT_EQ(d.memory_stats().cached_bytes, 40 * mb);
  // A hit: the same buffer, and no release.
  absl::StatusOr<Allocation> b = d.Allocate(40 * mb - 100);
  ASSERT_THAT(b, IsOk());
  EXPECT_EQ(b->ptr, a->ptr);
  EXPECT_EQ(d.memory_stats().cache_hits, 1u);
  ASSERT_THAT(d.Deallocate(b->ptr), IsOk());
  EXPECT_EQ(d.allocation_generation(), gen);
  // A miss that only fits once the cached 40 MB is evicted.
  absl::StatusOr<Allocation> c = d.Allocate(48 * mb);
  ASSERT_THAT(c, IsOk());
  Device::MemoryStats m = d.memory_stats();
  EXPECT_EQ(m.live_bytes, 48 * mb);
  EXPECT_EQ(m.cached_bytes, 0u);
  EXPECT_GT(d.allocation_generation(), gen);
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
  EXPECT_THAT(dev_->error(), StatusIs(absl::StatusCode::kInternal,
                                      HasSubstr("restart")));
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
  EXPECT_EQ(dev_->unsignaled_host_task_waits_committed(), 0u);
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
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(dev_->Resolve(&host),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(dev_->CompileLibrary("kernel void broken("),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("domain=")));
  EXPECT_THAT(dev_->CreateKernel(lib_, "no_such_function"),
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
  absl::StatusOr<MTL::Library*> lib = dev_->CompileLibrary(msl);
  ASSERT_THAT(lib, IsOk());
  absl::StatusOr<std::unique_ptr<Kernel>> k =
      dev_->CreateKernel(*lib, "sum_many");
  ASSERT_THAT(k, IsOk());
  (*k)->set_uses_argument_buffer(true);

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
  absl::StatusOr<MTL::Library*> lib = dev_->CompileLibrary(msl);
  ASSERT_THAT(lib, IsOk());
  absl::StatusOr<std::unique_ptr<Kernel>> k = dev_->CreateKernel(*lib, "capped");
  ASSERT_THAT(k, IsOk());
  RecordProperty("pso_max_threads_with_attribute_64",
                 static_cast<int>((*k)->max_total_threads_per_threadgroup()));
  // Even if the pipeline reported more, the declared value is the limit.
  (*k)->set_declared_max_threads(64);
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

TEST_F(MetalRuntimeTest, MemoryBudgetAndAllocationGuard) {
  // The budget is derived from free system memory and capped by the working
  // set; it must be usable but never the whole machine.
  EXPECT_GE(dev_->memory_budget(), 256ull << 20);
  EXPECT_LE(dev_->memory_budget(), dev_->info().recommended_working_set);
  EXPECT_LE(dev_->memory_budget(), PhysicalMemoryBytes() / 2);
  // Asking for more than the machine can hand out without swapping is
  // refused cleanly (RESOURCE_EXHAUSTED), not attempted.
  uint64_t too_much = std::min<uint64_t>(dev_->info().max_buffer_length,
                                         PhysicalMemoryBytes());
  EXPECT_THAT(dev_->Allocate(too_much),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(dev_->allocated_bytes(), 0);
}

}  // namespace
}  // namespace metal_pjrt::rt
