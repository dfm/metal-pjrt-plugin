// Standalone smoke test for the runtime layer (no XLA, no gtest).
#include "metal_pjrt_plugin/runtime/metal_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace metal_pjrt::rt;

#define CHECK_OK(expr)                                                      \
  do {                                                                      \
    Status _s = (expr);                                                     \
    if (!_s.ok()) {                                                         \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,               \
              _s.message().c_str());                                        \
      return 1;                                                             \
    }                                                                       \
  } while (0)

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (!(cond)) {                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
      return 1;                                                        \
    }                                                                  \
  } while (0)

static const char* kMsl = R"(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; float alpha; };
kernel void axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 constant Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) y[i] = p.alpha * x[i] + y[i];
}
)";

int main() {
  printf("visible devices: %d\n", Device::VisibleDeviceCount());
  std::unique_ptr<Device> dev;
  CHECK_OK(Device::Create(0, &dev));
  const DeviceInfo& info = dev->info();
  printf("device %s family=%d metal4=%d maxbuf=%lluMB tgmem=%u\n", info.name.c_str(),
         info.gpu_family, info.supports_metal4,
         (unsigned long long)info.max_buffer_length >> 20, info.threadgroup_memory_length);

  MTL::Library* lib = nullptr;
  CHECK_OK(dev->CompileLibrary(kMsl, &lib));
  MTL::Library* lib2 = nullptr;
  CHECK_OK(dev->CompileLibrary(kMsl, &lib2));
  EXPECT(lib == lib2, "library cache miss");
  std::unique_ptr<Kernel> k;
  CHECK_OK(dev->CreateKernel(lib, "axpy", &k));
  printf("kernel simd width=%u max tg=%u\n", k->thread_execution_width(),
         k->max_total_threads_per_threadgroup());

  const uint32_t n = 1 << 20;
  Allocation x, y, z;
  CHECK_OK(dev->Allocate(n * 4, &x));
  CHECK_OK(dev->Allocate(n * 4, &y));
  CHECK_OK(dev->Allocate(n * 4, &z));
  // Interior pointer resolution.
  BufferRef ref;
  CHECK_OK(dev->Resolve(static_cast<char*>(y.ptr) + 4096, &ref));
  EXPECT(ref.offset == 4096, "interior offset");

  std::unique_ptr<Stream> s;
  CHECK_OK(dev->CreateStream(&s));
  std::vector<float> hx(n, 1.0f), hy(n, 2.0f), out(n, 0.0f);
  CHECK_OK(s->MemcpyHostToDevice(x.ptr, hx.data(), n * 4));
  CHECK_OK(s->MemcpyHostToDevice(y.ptr, hy.data(), n * 4));
  struct { uint32_t n; float alpha; } params = {n, 3.0f};
  // Launch 10 times: y += 3x each time -> y = 2 + 30 = 32.
  for (int i = 0; i < 10; ++i) {
    CHECK_OK(s->Launch(*k, Dim3{(n + 255) / 256, 1, 1}, Dim3{256, 1, 1},
                       {KernelArg::Buffer(x.ptr), KernelArg::Buffer(y.ptr),
                        KernelArg::Bytes(&params, sizeof(params))}));
  }
  // D2D copy then D2H, all ordered on the stream.
  CHECK_OK(s->MemcpyDeviceToDevice(z.ptr, y.ptr, n * 4));
  CHECK_OK(s->Memset8(y.ptr, 0, n * 4));
  CHECK_OK(s->MemcpyDeviceToHost(out.data(), z.ptr, n * 4));
  CHECK_OK(s->Synchronize());
  int bad = 0;
  for (uint32_t i = 0; i < n; ++i) if (out[i] != 32.0f) ++bad;
  printf("out[0]=%g y[0]=%g z[0]=%g bad=%d\n", out[0], static_cast<float*>(y.ptr)[0], static_cast<float*>(z.ptr)[0], bad);
  EXPECT(bad == 0, "axpy chain result wrong");
  EXPECT(static_cast<float*>(y.ptr)[123] == 0.0f, "memset8 failed");

  // Events across two streams: s2 waits for s's work.
  std::unique_ptr<Stream> s2;
  CHECK_OK(dev->CreateStream(&s2));
  std::unique_ptr<Event> ev;
  CHECK_OK(dev->CreateEvent(&ev));
  params.alpha = 1.0f;
  CHECK_OK(s->Launch(*k, Dim3{(n + 255) / 256, 1, 1}, Dim3{256, 1, 1},
                     {KernelArg::Buffer(x.ptr), KernelArg::Buffer(y.ptr),
                      KernelArg::Bytes(&params, sizeof(params))}));  // y = 1
  CHECK_OK(s->RecordEvent(ev.get()));
  CHECK_OK(s2->WaitForEvent(ev.get()));
  CHECK_OK(s2->Launch(*k, Dim3{(n + 255) / 256, 1, 1}, Dim3{256, 1, 1},
                      {KernelArg::Buffer(x.ptr), KernelArg::Buffer(y.ptr),
                       KernelArg::Bytes(&params, sizeof(params))}));  // y = 2
  CHECK_OK(s2->Synchronize());
  EXPECT(ev->IsComplete(), "event not complete");
  EXPECT(static_cast<float*>(y.ptr)[777] == 2.0f, "cross-stream ordering wrong");

  // Host callback ordering + Memset32.
  int calls = 0;
  CHECK_OK(s->HostCallback([&calls]() { ++calls; }));
  CHECK_OK(s->Memset32(z.ptr, 0x3f800000u, n * 4));  // 1.0f pattern
  CHECK_OK(s->Synchronize());
  EXPECT(calls == 1, "host callback did not run");
  EXPECT(static_cast<float*>(z.ptr)[5] == 1.0f, "memset32 failed");

  // Many small launches to exercise command-buffer rollover (>64 ops).
  for (int i = 0; i < 200; ++i) {
    CHECK_OK(s->Launch(*k, Dim3{4, 1, 1}, Dim3{256, 1, 1},
                       {KernelArg::Buffer(x.ptr), KernelArg::Buffer(y.ptr),
                        KernelArg::Bytes(&params, sizeof(params))}));
  }
  CHECK_OK(s->Synchronize());
  EXPECT(static_cast<float*>(y.ptr)[0] == 202.0f, "rollover chain wrong");

  CHECK_OK(dev->Deallocate(x.ptr));
  CHECK_OK(dev->Deallocate(y.ptr));
  CHECK_OK(dev->Deallocate(z.ptr));
  EXPECT(dev->allocated_bytes() == 0, "leak");
  printf("ALL OK\n");
  return 0;
}
