// Tests for the runtime layer on its own (no XLA). Needs a Metal device.
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/runtime/system_memory.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <string>
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
  ASSERT_THAT(s->HostCallback([&calls]() { ++calls; }), IsOk());
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
