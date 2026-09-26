// The plugin's exported GetPjrtApi: XLA's GPU C API shim with the
// PJRT_AbiVersion extension removed.
//
// With that extension, PjRtCApiExecutable::GetAbiVersion reports the OneAPI
// executable ABI, which jaxlib's IFRT (GetXlaExecutableVersion) rejects
// ("Unsupported platform ID for XlaExecutableAbiVersion"), so executables
// could not be serialized and JAX's persistent compilation cache could not
// work. Without it, GetAbiVersion is Unimplemented and IFRT takes the
// version-less path, as for CPU. The cache key's platform_version
// ("oneapi 1.<build>.<settings>", MetalExecutor::PluginVersion) is what keeps
// executables from another build or other compile-time settings out.

#include <cstdlib>
#include <cstring>

#include "absl/log/initialize.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "xla/pjrt/c/pjrt_c_api_macros.h"

extern "C" PJRT_CAPI_EXPORT const PJRT_Api* GetPjrtApi() {
  static const PJRT_Api* api = [] {
    // XLA code logs through absl (as XLA's own GetPjrtApi does).
    absl::InitializeLog();
    const PJRT_Api* gpu = pjrt::gpu_plugin::GetGpuPjrtApi();
    static PJRT_Api copy = *gpu;
    // Copy every other extension node (they are plain structs of function
    // pointers), relinked in the same order. Allocated once, never freed.
    PJRT_Extension_Base** tail = &copy.extension_start;
    for (PJRT_Extension_Base* ext = gpu->extension_start; ext != nullptr;
         ext = ext->next) {
      if (ext->type == PJRT_Extension_Type_AbiVersion) continue;
      auto* node =
          static_cast<PJRT_Extension_Base*>(std::malloc(ext->struct_size));
      std::memcpy(node, ext, ext->struct_size);
      *tail = node;
      tail = &node->next;
    }
    *tail = nullptr;
    return &copy;
  }();
  return api;
}
