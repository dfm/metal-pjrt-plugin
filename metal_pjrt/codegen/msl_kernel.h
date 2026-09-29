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

// The first line of every emitted kernel, standing for the MSL prelude
// (kernels/msl_prelude.metal, ~9 KB): the runtime puts the prelude back
// before compiling. XLA keeps each kernel thunk's source twice (the thunk and
// the executable's serialized thunks), one copy per thunk even when kernels
// are shared, so the prelude would be most of an executable's MSL. Must match
// metal_pjrt::rt::kMslPreludeLine (runtime/metal_runtime.h).
inline constexpr char kMslPreludeLine[] =
    "#include <metal_pjrt/msl_prelude.metal>\n";

struct MslKernel {
  // Name of the `kernel` function in `msl_source` (== the XLA entry function).
  std::string kernel_name;
  // An MSL translation unit, complete once its first line, kMslPreludeLine,
  // is replaced by the prelude (rt::Device::GetKernel does that).
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
