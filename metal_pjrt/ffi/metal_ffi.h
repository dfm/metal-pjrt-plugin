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
#ifndef METAL_PJRT_FFI_METAL_FFI_H_
#define METAL_PJRT_FFI_METAL_FFI_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"
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

// MSL scalar type name for an XLA element type ("float", "half", "bfloat",
// "int"), or an error for unsupported types.
absl::StatusOr<std::string> MslTypeName(xla::PrimitiveType type);

// Buffers + params launch of a compiled kernel (runtime/kernel_launch.h).
using rt::LaunchKernel;

// LaunchKernel of `function` from `msl_source`, compiled for the stream's
// device (cached there).
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
      ctx->device->GetKernel(msl_source, function);
  if (!kernel.ok()) return kernel.status();
  return LaunchKernel(ctx->stream, **kernel, buffers, params, threadgroups,
                      threads, threadgroup_memory_bytes);
}

}  // namespace ffi
}  // namespace metal_pjrt

#endif  // METAL_PJRT_FFI_METAL_FFI_H_
