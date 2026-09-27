// Helpers for XLA FFI custom-call handlers on the Metal platform.
//
// How a custom call reaches a handler here:
//  - The HLO custom call must use API_VERSION_TYPED_FFI. The thunk emitter
//    parses its backend_config (an MLIR dictionary) into FFI attributes and
//    creates a CustomCallThunk, which looks the handler up with
//    ffi::FindHandler(target, platform_name). platform_name is the SE
//    platform name "METAL"; the registry canonicalizes it (and the name given
//    at registration) through PlatformUtil::CanonicalPlatformName to "metal",
//    so "METAL" and "metal" are the same key.
//  - Handlers are registered statically inside the plugin dylib with
//    XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), name, "METAL", h).
//    The FFI registry is a static in the dylib's copy of XLA, so nothing is
//    needed from Python: jax.ffi.ffi_call(name, ...) emits the custom call and
//    the plugin's compiler/runtime find the handler.
//  - Bind the stream with .Ctx<xla::ffi::Stream>() (se::Stream*, from
//    xla/backends/gpu/ffi.h). MetalStream::platform_specific_handle().stream
//    is the metal_pjrt::rt::Stream*, and stream->parent() is the
//    MetalExecutor, which owns the rt::Device.
//
// Buffers (ffi::AnyBuffer::untyped_data()) are raw device pointers into
// shared MTLBuffers; rt::Stream::Launch resolves them.
#ifndef METAL_PJRT_PLUGIN_FFI_METAL_FFI_H_
#define METAL_PJRT_PLUGIN_FFI_METAL_FFI_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace metal_pjrt {
namespace ffi {

// SE platform name used for registration (canonicalized to "metal").
inline constexpr char kMetalFfiPlatform[] = "METAL";

// Custom-call target names of the handlers in this directory.
inline constexpr char kScanTarget[] = "metal$scan";
// XLA's SortRewriter targets (xla/service/gpu/cublas_cudnn.h).
inline constexpr char kCubSortKeysTarget[] = "xla.gpu.ext.cub_sort_keys";
inline constexpr char kCubSortPairsTarget[] = "xla.gpu.ext.cub_sort_pairs";

// The runtime stream and device behind an se::Stream owned by the Metal
// StreamExecutor. Fails for streams of other platforms.
struct MetalContext {
  rt::Stream* stream = nullptr;
  rt::Device* device = nullptr;
};
absl::StatusOr<MetalContext> GetMetalContext(stream_executor::Stream* stream);

// The runtime device of Metal executor 0 (the only one). For the FFI
// instantiate stage, which runs without a stream.
absl::StatusOr<rt::Device*> DefaultMetalDevice();

// Compiles `msl_source` (once per device and source) and returns the kernel
// `function` from it. The returned pointer lives as long as the process.
absl::StatusOr<const rt::Kernel*> GetOrCreateKernel(
    rt::Device* device, const std::string& msl_source,
    const std::string& function);

// MSL scalar type name for an XLA element type ("float", "half", "bfloat",
// "int"), or an error for unsupported types.
absl::StatusOr<std::string> MslTypeName(xla::PrimitiveType type);

// Launches `function` from `msl_source` on the stream with the given buffers
// in [[buffer(0..n-1)]] followed by `params` as setBytes in [[buffer(n)]].
// `Params` must be trivially copyable (a plain struct matching the MSL one).
// A 1-D threadgroup (threads.y == threads.z == 1) larger than the pipeline's
// maxTotalThreadsPerThreadgroup is clamped to it, rounded down to a multiple
// of the SIMD width, so kernels must loop with a threadgroup-size stride.
// LaunchKernel is the same for an already compiled kernel.
template <typename Params>
absl::Status LaunchKernel(rt::Stream* stream, const rt::Kernel& kernel,
                          const std::vector<const void*>& buffers,
                          const Params& params, rt::Dim3 threadgroups,
                          rt::Dim3 threads,
                          uint32_t threadgroup_memory_bytes = 0) {
  static_assert(std::is_trivially_copyable_v<Params>);
  if (threads.y == 1 && threads.z == 1) {
    uint32_t max_threads = kernel.max_total_threads_per_threadgroup();
    uint32_t width = std::max<uint32_t>(kernel.thread_execution_width(), 1);
    if (max_threads > 0 && threads.x > max_threads) {
      threads.x = std::max(width, max_threads / width * width);
    }
  }
  std::vector<rt::KernelArg> args;
  args.reserve(buffers.size() + 1);
  for (const void* b : buffers) args.push_back(rt::KernelArg::Buffer(b));
  args.push_back(rt::KernelArg::Bytes(&params, sizeof(Params)));
  return stream->Launch(kernel, threadgroups, threads, args,
                        threadgroup_memory_bytes);
}

template <typename Params>
absl::Status LaunchMsl(stream_executor::Stream* stream,
                       const std::string& msl_source,
                       const std::string& function,
                       const std::vector<const void*>& buffers,
                       const Params& params, rt::Dim3 threadgroups,
                       rt::Dim3 threads,
                       uint32_t threadgroup_memory_bytes = 0) {
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  absl::StatusOr<const rt::Kernel*> kernel =
      GetOrCreateKernel(ctx->device, msl_source, function);
  if (!kernel.ok()) return kernel.status();
  return LaunchKernel(ctx->stream, **kernel, buffers, params, threadgroups,
                      threads, threadgroup_memory_bytes);
}

}  // namespace ffi
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_FFI_METAL_FFI_H_
