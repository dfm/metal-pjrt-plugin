// The plugin's exported GetPjrtApi: XLA's GPU C API shim with the
// PJRT_AbiVersion extension removed, and PJRT_Client_BufferFromHostBuffer
// wrapped so the caller's host buffer is copied before it returns (below).
//
// With that extension, PjRtCApiExecutable::GetAbiVersion reports the OneAPI
// executable ABI, which jaxlib's IFRT (GetXlaExecutableVersion) rejects
// ("Unsupported platform ID for XlaExecutableAbiVersion"), so executables
// could not be serialized and JAX's persistent compilation cache could not
// work. Without it, GetAbiVersion is Unimplemented and IFRT takes the
// version-less path, as for CPU. The cache key's platform_version
// ("oneapi 1.<build>.<settings>", MetalExecutor::PluginVersion) is what keeps
// executables from another build or other compile-time settings out.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "absl/log/initialize.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_macros.h"
#include "xla/primitive_util.h"

namespace {

const PJRT_Api* g_gpu = nullptr;  // XLA's shim

// jaxlib passes numpy arrays with semantics that let the H2D copy read them
// after the call returns (XLA dispatches it on a worker thread), so a
// caller refilling its array right after device_put (a data loader) changed
// what the device got. Copy dense host data here instead, as CUDA does for
// pageable memory, and free the copy when XLA is done with it. Other
// layouts (strided, sub-byte types) take XLA's synchronous path
// (kImmutableOnlyDuringCall).
PJRT_Error* BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  if (args->host_buffer_semantics ==
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall) {
    return g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  }
  const xla::PrimitiveType type = pjrt::ConvertFromPjRtBufferType(args->type);
  const bool dense = args->num_byte_strides == 0 &&
                     type != xla::TOKEN && xla::primitive_util::IsArrayType(type) &&
                     !xla::primitive_util::IsSubByteNonPredType(type);
  if (!dense) {
    args->host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    return g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  }
  uint64_t size = xla::primitive_util::ByteWidth(type);
  for (size_t i = 0; i < args->num_dims; ++i) size *= args->dims[i];
  void* copy = std::malloc(size == 0 ? 1 : size);
  if (copy == nullptr) {
    args->host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    return g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  }
  std::memcpy(copy, args->data, size);
  const void* data = args->data;
  args->data = copy;
  args->host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  PJRT_Error* error = g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  args->data = data;
  if (error != nullptr) {
    std::free(copy);
    return error;
  }
  PJRT_Event_OnReady_Args on_ready;
  on_ready.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  on_ready.extension_start = nullptr;
  on_ready.event = args->done_with_host_buffer;
  on_ready.user_arg = copy;
  on_ready.callback = [](PJRT_Error* e, void* user_arg) {
    std::free(user_arg);
    if (e != nullptr) {
      PJRT_Error_Destroy_Args d{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr,
                                e};
      g_gpu->PJRT_Error_Destroy(&d);
    }
  };
  return g_gpu->PJRT_Event_OnReady(&on_ready);
}

}  // namespace

extern "C" PJRT_CAPI_EXPORT const PJRT_Api* GetPjrtApi() {
  static const PJRT_Api* api = [] {
    // XLA code logs through absl (as XLA's own GetPjrtApi does).
    absl::InitializeLog();
    const PJRT_Api* gpu = pjrt::gpu_plugin::GetGpuPjrtApi();
    g_gpu = gpu;
    static PJRT_Api copy = *gpu;
    copy.PJRT_Client_BufferFromHostBuffer = BufferFromHostBuffer;
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
