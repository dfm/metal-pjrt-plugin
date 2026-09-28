// The plugin's exported GetPjrtApi: XLA's GPU C API shim with the
// PJRT_AbiVersion extension removed, PJRT_Client_BufferFromHostBuffer
// wrapped so the caller's host buffer is copied (or, when large, done with)
// before it returns, and
// PJRT_Error_Message wrapped so out-of-memory errors say why (below).
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
#include <mutex>
#include <string>
#include <unordered_map>

#include "absl/log/initialize.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_macros.h"
#include "xla/primitive_util.h"
#include "tsl/platform/numbers.h"

namespace {

const PJRT_Api* g_gpu = nullptr;  // XLA's shim

// jaxlib passes numpy arrays with semantics that let the H2D copy read them
// after the call returns (XLA dispatches it on a worker thread), so a
// caller refilling its array right after device_put (a data loader) changed
// what the device got. Below kSnapshotLimit, copy dense host data here
// instead, as CUDA does for pageable memory, and free the copy when XLA is
// done with it. At or above it, a copy would double the host footprint, so
// hand XLA the caller's data and wait until XLA is done with it (the copy
// ran, or failed on a device error) before returning. The wait needs no
// Python: jaxlib calls this without the GIL, and the copy runs on XLA's
// worker threads and the runtime's host-to-device stream. It does wait for
// GPU work already queued (the copy waits for XLA's allocation event on the
// compute stream). jaxlib passes explicit strides even for C-contiguous
// arrays; without IsDense accepting them every array took the staged
// synchronous path, which is what doubled the footprint. Other layouts
// (strided, sub-byte types) take XLA's synchronous path
// (kImmutableOnlyDuringCall), which linearizes into a host staging copy.
constexpr uint64_t kSnapshotLimit = uint64_t{16} << 20;

// Row-major and unpadded: no strides, or the strides jaxlib passes for a
// C-contiguous numpy array (those of size-1 dimensions do not matter).
bool IsDense(const PJRT_Client_BufferFromHostBuffer_Args& args,
             xla::PrimitiveType type) {
  if (type == xla::TOKEN || !xla::primitive_util::IsArrayType(type) ||
      xla::primitive_util::IsSubByteNonPredType(type)) {
    return false;
  }
  if (args.num_byte_strides == 0) return true;
  if (args.num_byte_strides != args.num_dims) return false;
  int64_t stride = xla::primitive_util::ByteWidth(type);
  for (size_t i = args.num_dims; i-- > 0;) {
    if (args.dims[i] != 1 && args.byte_strides[i] != stride) return false;
    stride *= args.dims[i];
  }
  return true;
}

PJRT_Error* BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  if (args->host_buffer_semantics ==
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall) {
    return g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  }
  const xla::PrimitiveType type = pjrt::ConvertFromPjRtBufferType(args->type);
  if (!IsDense(*args, type)) {
    args->host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    return g_gpu->PJRT_Client_BufferFromHostBuffer(args);
  }
  uint64_t size = xla::primitive_util::ByteWidth(type);
  for (size_t i = 0; i < args->num_dims; ++i) size *= args->dims[i];
  if (size >= kSnapshotLimit) {
    args->host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    PJRT_Error* error = g_gpu->PJRT_Client_BufferFromHostBuffer(args);
    if (error != nullptr) return error;
    // The event stays the caller's; awaiting it does not consume it. Its
    // error (if any) also reaches the buffer, so it is not ours to return.
    PJRT_Event_Await_Args await;
    await.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await.extension_start = nullptr;
    await.event = args->done_with_host_buffer;
    if (PJRT_Error* e = g_gpu->PJRT_Event_Await(&await); e != nullptr) {
      PJRT_Error_Destroy_Args d{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr,
                                e};
      g_gpu->PJRT_Error_Destroy(&d);
    }
    return nullptr;
  }
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

// XLA's allocator adapter replaces the runtime's refusal (budget or system
// memory guard, with numbers) by "Out of memory while trying to allocate
// <size> with allocator ...". Append the runtime's reason when the last
// refusal was of that size. The annotated text lives until the error is
// destroyed, as the C API requires.
std::mutex g_messages_mu;
std::unordered_map<const PJRT_Error*, std::string>* g_messages =
    new std::unordered_map<const PJRT_Error*, std::string>;

void ErrorMessage(PJRT_Error_Message_Args* args) {
  g_gpu->PJRT_Error_Message(args);
  const absl::string_view message(args->message, args->message_size);
  if (!absl::StartsWith(message, "Out of memory while trying to allocate ")) {
    return;
  }
  uint64_t size = 0;
  const std::string reason = metal_pjrt::rt::LastAllocationRefusal(&size);
  if (reason.empty() ||
      !absl::StrContains(message,
                         absl::StrCat(" allocate ",
                                      tsl::strings::HumanReadableNumBytes(size),
                                      " "))) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_messages_mu);
  auto [it, inserted] = g_messages->try_emplace(args->error);
  if (inserted) it->second = absl::StrCat(message, " openmetal: ", reason);
  args->message = it->second.data();
  args->message_size = it->second.size();
}

void ErrorDestroy(PJRT_Error_Destroy_Args* args) {
  {
    std::lock_guard<std::mutex> lock(g_messages_mu);
    g_messages->erase(args->error);
  }
  g_gpu->PJRT_Error_Destroy(args);
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
    copy.PJRT_Error_Message = ErrorMessage;
    copy.PJRT_Error_Destroy = ErrorDestroy;
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
