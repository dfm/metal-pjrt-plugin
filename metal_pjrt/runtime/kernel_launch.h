// LaunchKernel: a compiled kernel with buffers plus a params struct, the
// calling convention of the hand-written kernels (kernels/*.metal). No XLA,
// so the dispatch libraries built on it (ffi/, linalg/) can be tested
// against rt::Device directly.
#ifndef METAL_PJRT_RUNTIME_KERNEL_LAUNCH_H_
#define METAL_PJRT_RUNTIME_KERNEL_LAUNCH_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace rt {

// Launches `kernel` on the stream with the given buffers in
// [[buffer(0..n-1)]] followed by `params` as setBytes in [[buffer(n)]].
// `Params` must be trivially copyable (a plain struct matching the MSL one).
// A 1-D threadgroup (threads.y == threads.z == 1) larger than the pipeline's
// maxTotalThreadsPerThreadgroup is clamped to it, rounded down to a multiple
// of the SIMD width, so kernels must loop with a threadgroup-size stride.
template <typename Params>
absl::Status LaunchKernel(Stream* stream, const Kernel& kernel,
                          const std::vector<const void*>& buffers,
                          const Params& params, Dim3 threadgroups,
                          Dim3 threads,
                          uint32_t threadgroup_memory_bytes = 0) {
  static_assert(std::is_trivially_copyable_v<Params>);
  if (threads.y == 1 && threads.z == 1) {
    uint32_t max_threads = kernel.max_total_threads_per_threadgroup();
    uint32_t width = std::max<uint32_t>(kernel.thread_execution_width(), 1);
    if (max_threads > 0 && threads.x > max_threads) {
      threads.x = std::max(width, max_threads / width * width);
    }
  }
  std::vector<KernelArg> args;
  args.reserve(buffers.size() + 1);
  for (const void* b : buffers) args.push_back(KernelArg::Buffer(b));
  args.push_back(KernelArg::Bytes(&params, sizeof(Params)));
  return stream->Launch(kernel, threadgroups, threads, args,
                        threadgroup_memory_bytes);
}

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_KERNEL_LAUNCH_H_
