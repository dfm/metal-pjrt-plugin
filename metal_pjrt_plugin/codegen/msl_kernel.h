#ifndef METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_H_
#define METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_H_

#include <string>

namespace metal_pjrt::codegen {

struct MslKernel {
  // Name of the `kernel` function in `msl_source` (== the XLA entry function).
  std::string kernel_name;
  // Complete, self-contained MSL translation unit.
  std::string msl_source;
  // The kernel takes exactly this many `device char*` arguments bound to
  // [[buffer(0)]] .. [[buffer(num_buffer_args - 1)]], in XLA's kernel argument
  // order (fusion operands followed by fusion results), followed by the
  // thread-position attributes.
  int num_buffer_args = 0;
};

}  // namespace metal_pjrt::codegen

#endif  // METAL_PJRT_PLUGIN_CODEGEN_MSL_KERNEL_H_
