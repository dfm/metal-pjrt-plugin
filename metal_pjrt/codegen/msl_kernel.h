#ifndef METAL_PJRT_CODEGEN_MSL_KERNEL_H_
#define METAL_PJRT_CODEGEN_MSL_KERNEL_H_

#include <string>

namespace metal_pjrt::codegen {

// Kernels with at most this many buffer arguments bind them directly to
// [[buffer(i)]] (Metal's argument table has 31 slots). Larger kernels use the
// argument-buffer convention: [[buffer(0)]] is `constant ulong*`, one GPU
// address per argument, and the source starts the kernel with
// kArgumentBufferMarker. Must match metal_pjrt::rt::kArgumentBufferMarker /
// rt::Stream::kMaxBufferArgs (runtime/metal_runtime.h).
inline constexpr int kMaxDirectBufferArgs = 31;
inline constexpr char kArgumentBufferMarker[] = "// xla_metal_argbuffer";

struct MslKernel {
  // Name of the `kernel` function in `msl_source` (== the XLA entry function).
  std::string kernel_name;
  // Complete, self-contained MSL translation unit.
  std::string msl_source;
  // The kernel takes exactly this many `device char*` arguments bound to
  // [[buffer(0)]] .. [[buffer(num_buffer_args - 1)]], in XLA's kernel argument
  // order (fusion operands followed by fusion results), followed by the
  // thread-position attributes. Above kMaxDirectBufferArgs they arrive through
  // an argument buffer instead (see above).
  int num_buffer_args = 0;
};

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_CODEGEN_MSL_KERNEL_H_
