// Spike: is replaying a Metal indirect command buffer (ICB) cheaper per
// command than encoding the same dispatches directly? Needs a Metal device;
// run under scripts/device_lock.py:
//   bazel build //metal_pjrt_plugin/runtime:icb_spike
//   scripts/device_lock.py -- bazel-bin/metal_pjrt_plugin/runtime/icb_spike [n] [reps]
//
// Workload: n trivial dispatches (one 256-thread threadgroup, 3 buffers:
// x = x + one, in place on one of 8 accumulators), each ordered after the
// previous one. Every variant must leave each accumulator equal to the number
// of dispatches that touched it, which checks that ordering (barriers) held.
//
// Variants (CPU time is host time to build and commit; GPU time is
// GPUEndTime - GPUStartTime of the command buffer):
//   direct tracked    serial encoder, tracked buffers, setPipeline+3 setBuffer
//                     + dispatch per command (today's runtime path)
//   direct untracked  serial encoder, untracked buffers + useResources
//   icb record        record all commands into an ICB (pipeline + 3
//                     setKernelBuffer + dispatch + setBarrier per command)
//   icb replay        one executeCommandsInBuffer per command buffer
//   icb rebind        rewrite the 3 buffer bindings of every command (the cost
//                     of an Update that re-encodes arguments)
//   table record/replay  kernels read their 3 addresses from an address table
//                     (constant ulong* [[buffer(0)]]); each command binds the
//                     table at its own offset, so commands never change
//   table update      rewrite every address in the table (a memcpy)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace {

constexpr char kMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
kernel void add3(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* c [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  c[i] = a[i] + b[i];
}
kernel void add3_table(constant ulong* xla_args [[buffer(0)]],
                       uint i [[thread_position_in_grid]]) {
  device const float* a = (device const float*)xla_args[0];
  device const float* b = (device const float*)xla_args[1];
  device float* c = (device float*)xla_args[2];
  c[i] = a[i] + b[i];
}
)";

constexpr uint32_t kThreads = 256;
constexpr int kAccumulators = 8;

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void Die(const char* what, NS::Error* err = nullptr) {
  std::fprintf(stderr, "%s%s%s\n", what, err ? ": " : "",
               err ? err->localizedDescription()->utf8String() : "");
  std::exit(1);
}

struct Result {
  double cpu_us_per_cmd;
  double gpu_us_per_cmd;
};

double GpuSeconds(MTL::CommandBuffer* cb) {
  if (cb->status() == MTL::CommandBufferStatusError) {
    Die("command buffer failed", cb->error());
  }
  return cb->GPUEndTime() - cb->GPUStartTime();
}

class Spike {
 public:
  explicit Spike(int n) : n_(n) {
    dev_ = MTL::CreateSystemDefaultDevice();
    if (!dev_) Die("no Metal device");
    queue_ = dev_->newCommandQueue();
    NS::Error* err = nullptr;
    MTL::Library* lib = dev_->newLibrary(
        NS::String::string(kMsl, NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) Die("newLibrary", err);
    pso_ = MakePso(lib, "add3");
    pso_table_ = MakePso(lib, "add3_table");
    auto opts_tracked = MTL::ResourceStorageModeShared |
                        MTL::ResourceHazardTrackingModeTracked;
    auto opts_untracked = MTL::ResourceStorageModeShared |
                          MTL::ResourceHazardTrackingModeUntracked;
    one_ = dev_->newBuffer(kThreads * sizeof(float), opts_tracked);
    one_u_ = dev_->newBuffer(kThreads * sizeof(float), opts_untracked);
    for (MTL::Buffer* b : {one_, one_u_}) {
      float* p = static_cast<float*>(b->contents());
      for (uint32_t j = 0; j < kThreads; ++j) p[j] = 1.0f;
    }
    for (int k = 0; k < kAccumulators; ++k) {
      acc_.push_back(dev_->newBuffer(kThreads * sizeof(float), opts_tracked));
      acc_u_.push_back(
          dev_->newBuffer(kThreads * sizeof(float), opts_untracked));
    }
    table_ = dev_->newBuffer(uint64_t(n_) * 3 * sizeof(uint64_t),
                             opts_untracked);
    resident_.push_back(one_u_);
    for (MTL::Buffer* b : acc_u_) resident_.push_back(b);
    resident_.push_back(table_);
    lib->release();
  }

  MTL::ComputePipelineState* MakePso(MTL::Library* lib, const char* name) {
    MTL::Function* fn =
        lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) Die("newFunction");
    MTL::ComputePipelineDescriptor* d =
        MTL::ComputePipelineDescriptor::alloc()->init();
    d->setComputeFunction(fn);
    d->setSupportIndirectCommandBuffers(true);
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = dev_->newComputePipelineState(
        d, MTL::PipelineOptionNone, nullptr, &err);
    if (!pso) Die("newComputePipelineState", err);
    d->release();
    fn->release();
    return pso;
  }

  void Zero(const std::vector<MTL::Buffer*>& bufs) {
    for (MTL::Buffer* b : bufs) std::memset(b->contents(), 0, b->length());
  }

  void Verify(const std::vector<MTL::Buffer*>& bufs, int multiplier,
              const char* what) {
    for (int k = 0; k < kAccumulators; ++k) {
      int expected = 0;
      for (int i = 0; i < n_; ++i) expected += (i % kAccumulators == k);
      expected *= multiplier;
      const float* p = static_cast<const float*>(bufs[k]->contents());
      for (uint32_t j = 0; j < kThreads; ++j) {
        if (p[j] != float(expected)) {
          std::fprintf(stderr, "%s: accumulator %d[%u] = %g, expected %d\n",
                       what, k, j, p[j], expected);
          std::exit(1);
        }
      }
    }
  }

  // (a) Direct encoding, as rt::Stream::Launch does today.
  Result Direct(bool tracked) {
    const std::vector<MTL::Buffer*>& acc = tracked ? acc_ : acc_u_;
    MTL::Buffer* one = tracked ? one_ : one_u_;
    Zero(acc);
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    double t0 = Now();
    MTL::CommandBuffer* cb = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* e =
        cb->computeCommandEncoder(MTL::DispatchTypeSerial);
    if (!tracked) {
      e->useResources(
          reinterpret_cast<const MTL::Resource* const*>(resident_.data()),
          resident_.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    for (int i = 0; i < n_; ++i) {
      MTL::Buffer* x = acc[i % kAccumulators];
      e->setComputePipelineState(pso_);
      e->setBuffer(x, 0, 0);
      e->setBuffer(one, 0, 1);
      e->setBuffer(x, 0, 2);
      e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(kThreads, 1, 1));
      // Untracked: the serial dispatch type alone orders the dispatches.
    }
    e->endEncoding();
    cb->commit();
    double t1 = Now();
    cb->waitUntilCompleted();
    Result r{(t1 - t0) * 1e6 / n_, GpuSeconds(cb) * 1e6 / n_};
    pool->release();
    Verify(acc, 1, tracked ? "direct tracked" : "direct untracked");
    return r;
  }

  // (a'') Direct encoding of the address-table kernel (isolates the GPU cost
  // of reading addresses from the table from the cost of the ICB).
  Result DirectTable() {
    WriteTable();
    Zero(acc_u_);
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    double t0 = Now();
    MTL::CommandBuffer* cb = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* e =
        cb->computeCommandEncoder(MTL::DispatchTypeSerial);
    e->useResources(
        reinterpret_cast<const MTL::Resource* const*>(resident_.data()),
        resident_.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    for (int i = 0; i < n_; ++i) {
      e->setComputePipelineState(pso_table_);
      e->setBuffer(table_, uint64_t(i) * 3 * sizeof(uint64_t), 0);
      e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(kThreads, 1, 1));
    }
    e->endEncoding();
    cb->commit();
    double t1 = Now();
    cb->waitUntilCompleted();
    Result r{(t1 - t0) * 1e6 / n_, GpuSeconds(cb) * 1e6 / n_};
    pool->release();
    Verify(acc_u_, 1, "direct table");
    return r;
  }

  MTL::IndirectCommandBuffer* NewIcb(int max_bind) {
    MTL::IndirectCommandBufferDescriptor* d =
        MTL::IndirectCommandBufferDescriptor::alloc()->init();
    d->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
    d->setInheritBuffers(false);
    d->setInheritPipelineState(false);
    d->setMaxKernelBufferBindCount(max_bind);
    MTL::IndirectCommandBuffer* icb = dev_->newIndirectCommandBuffer(
        d, n_, MTL::ResourceStorageModePrivate);
    d->release();
    if (!icb) Die("newIndirectCommandBuffer");
    return icb;
  }

  // (b)/(c) ICB with per-command buffer bindings.
  void IcbBindings(Result* record, Result* replay, double* rebind_us) {
    MTL::IndirectCommandBuffer* icb = NewIcb(3);
    double t0 = Now();
    for (int i = 0; i < n_; ++i) {
      MTL::IndirectComputeCommand* c = icb->indirectComputeCommand(i);
      MTL::Buffer* x = acc_u_[i % kAccumulators];
      c->setComputePipelineState(pso_);
      c->setKernelBuffer(x, 0, 0);
      c->setKernelBuffer(one_u_, 0, 1);
      c->setKernelBuffer(x, 0, 2);
      c->concurrentDispatchThreadgroups(MTL::Size(1, 1, 1),
                                        MTL::Size(kThreads, 1, 1));
      c->setBarrier();
    }
    double t1 = Now();
    *record = {(t1 - t0) * 1e6 / n_, 0};
    *replay = Replay(icb, "icb replay");
    // Update: rebind every command's buffers (same values; the cost is the
    // same as binding new ones).
    t0 = Now();
    for (int i = 0; i < n_; ++i) {
      MTL::IndirectComputeCommand* c = icb->indirectComputeCommand(i);
      MTL::Buffer* x = acc_u_[i % kAccumulators];
      c->setKernelBuffer(x, 0, 0);
      c->setKernelBuffer(one_u_, 0, 1);
      c->setKernelBuffer(x, 0, 2);
    }
    t1 = Now();
    *rebind_us = (t1 - t0) * 1e6 / n_;
    Replay(icb, "icb replay after rebind");
    icb->release();
  }

  // (d) ICB whose commands bind the address table at a fixed offset.
  void IcbTable(Result* record, Result* replay, double* update_us) {
    MTL::IndirectCommandBuffer* icb = NewIcb(1);
    WriteTable();
    double t0 = Now();
    for (int i = 0; i < n_; ++i) {
      MTL::IndirectComputeCommand* c = icb->indirectComputeCommand(i);
      c->setComputePipelineState(pso_table_);
      c->setKernelBuffer(table_, uint64_t(i) * 3 * sizeof(uint64_t), 0);
      c->concurrentDispatchThreadgroups(MTL::Size(1, 1, 1),
                                        MTL::Size(kThreads, 1, 1));
      c->setBarrier();
    }
    double t1 = Now();
    *record = {(t1 - t0) * 1e6 / n_, 0};
    *replay = Replay(icb, "table replay");
    t0 = Now();
    WriteTable();
    t1 = Now();
    *update_us = (t1 - t0) * 1e6 / n_;
    Replay(icb, "table replay after update");
    icb->release();
  }

  void WriteTable() {
    uint64_t* t = static_cast<uint64_t*>(table_->contents());
    const uint64_t one = one_u_->gpuAddress();
    for (int i = 0; i < n_; ++i) {
      const uint64_t x = acc_u_[i % kAccumulators]->gpuAddress();
      t[3 * i + 0] = x;
      t[3 * i + 1] = one;
      t[3 * i + 2] = x;
    }
  }

  Result Replay(MTL::IndirectCommandBuffer* icb, const char* what) {
    Zero(acc_u_);
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    double t0 = Now();
    MTL::CommandBuffer* cb = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* e =
        cb->computeCommandEncoder(MTL::DispatchTypeSerial);
    e->useResources(
        reinterpret_cast<const MTL::Resource* const*>(resident_.data()),
        resident_.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    e->executeCommandsInBuffer(icb, NS::Range(0, n_));
    e->endEncoding();
    cb->commit();
    double t1 = Now();
    cb->waitUntilCompleted();
    Result r{(t1 - t0) * 1e6 / n_, GpuSeconds(cb) * 1e6 / n_};
    pool->release();
    Verify(acc_u_, 1, what);
    return r;
  }

 private:
  int n_;
  MTL::Device* dev_ = nullptr;
  MTL::CommandQueue* queue_ = nullptr;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::ComputePipelineState* pso_table_ = nullptr;
  MTL::Buffer* one_ = nullptr;
  MTL::Buffer* one_u_ = nullptr;
  MTL::Buffer* table_ = nullptr;
  std::vector<MTL::Buffer*> acc_, acc_u_, resident_;
};

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 10000;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  Spike spike(n);
  std::printf("%d dispatches per run; us per command (CPU = host build+commit, "
              "GPU = GPUEnd-GPUStart)\n", n);
  for (int r = 0; r < reps; ++r) {
    Result dt = spike.Direct(true);
    Result du = spike.Direct(false);
    Result dtab = spike.DirectTable();
    Result rec, rep, trec, trep;
    double rebind, tupdate;
    spike.IcbBindings(&rec, &rep, &rebind);
    spike.IcbTable(&trec, &trep, &tupdate);
    std::printf(
        "rep %d:\n"
        "  direct tracked    cpu %6.3f  gpu %6.3f\n"
        "  direct untracked  cpu %6.3f  gpu %6.3f\n"
        "  direct table      cpu %6.3f  gpu %6.3f\n"
        "  icb record        cpu %6.3f\n"
        "  icb replay        cpu %6.3f  gpu %6.3f\n"
        "  icb rebind (upd)  cpu %6.3f\n"
        "  table record      cpu %6.3f\n"
        "  table replay      cpu %6.3f  gpu %6.3f\n"
        "  table update      cpu %6.3f\n",
        r, dt.cpu_us_per_cmd, dt.gpu_us_per_cmd, du.cpu_us_per_cmd,
        du.gpu_us_per_cmd, dtab.cpu_us_per_cmd, dtab.gpu_us_per_cmd,
        rec.cpu_us_per_cmd, rep.cpu_us_per_cmd,
        rep.gpu_us_per_cmd, rebind, trec.cpu_us_per_cmd, trep.cpu_us_per_cmd,
        trep.gpu_us_per_cmd, tupdate);
  }
  pool->release();
  return 0;
}
