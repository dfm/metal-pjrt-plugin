// "xla_ffi_python_metal_callback": host (Python) callbacks on the Metal
// platform, used by jax.pure_callback, io_callback, jax.debug.callback and
// jax.debug.print (lowering in metal_pjrt_plugin/callbacks.py).
//
// Upstream JAX routes callbacks through FfiLoadedHostCallbacks user data that
// jaxlib only attaches for the cpu/cuda/rocm/oneapi platform ids, so it never
// reaches this plugin. Instead the callable is identified by a
// `callback_id` attribute and resolved by a Python trampoline that Python
// installs through the exported C function below (via ctypes; ctypes
// acquires the GIL when a foreign thread calls the trampoline).
//
// The handler waits for all prior work on the execution stream; buffers are
// shared-storage MTLBuffers whose "device pointers" are host addresses, so the
// trampoline reads arguments and writes results in place (row-major, the
// custom call's default layout). Token operands/results are skipped.
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

extern "C" {

// One argument or result as seen by the trampoline. `dtype` is the XLA
// PrimitiveType number; `data` is host-addressable (may be null when empty).
struct MetalPjrtCallbackBuffer {
  void* data;
  int64_t dtype;
  int64_t rank;
  const int64_t* dims;
};

// Returns 0 on success. On failure writes a NUL-terminated message (at most
// `error_capacity` bytes including the NUL) to `error` and returns nonzero.
typedef int (*MetalPjrtPythonCallbackTrampoline)(
    uint64_t callback_id, int64_t num_args, const MetalPjrtCallbackBuffer* args,
    int64_t num_results, const MetalPjrtCallbackBuffer* results, char* error,
    int64_t error_capacity);

__attribute__((visibility("default"))) void
metal_pjrt_register_python_callback_trampoline(
    MetalPjrtPythonCallbackTrampoline fn);

}  // extern "C"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

std::atomic<MetalPjrtPythonCallbackTrampoline> g_trampoline{nullptr};

absl::Status PythonCallback(stream_executor::Stream* stream,
                            uint64_t callback_id, xffi::RemainingArgs args,
                            xffi::RemainingRets rets) {
  MetalPjrtPythonCallbackTrampoline trampoline = g_trampoline.load();
  if (trampoline == nullptr) {
    return absl::FailedPreconditionError(
        "xla_ffi_python_metal_callback: no Python trampoline registered "
        "(metal_pjrt_plugin did not initialize host callbacks)");
  }
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  // Everything enqueued before this custom call must have finished before
  // the host reads its operands (and before it overwrites its results).
  if (absl::Status s = ctx->stream->Synchronize(); !s.ok()) return s;

  auto describe = [](const xffi::AnyBuffer& b) {
    MetalPjrtCallbackBuffer d;
    d.data = b.untyped_data();
    d.dtype = static_cast<int64_t>(b.element_type());
    auto dims = b.dimensions();
    d.rank = static_cast<int64_t>(dims.size());
    d.dims = dims.begin();
    return d;
  };
  std::vector<MetalPjrtCallbackBuffer> in, out;
  in.reserve(args.size());
  out.reserve(rets.size());
  for (size_t i = 0; i < args.size(); ++i) {
    absl::StatusOr<xffi::AnyBuffer> b = args.get<xffi::AnyBuffer>(i);
    if (!b.ok()) return b.status();
    if (b->element_type() == xla::TOKEN) continue;
    in.push_back(describe(*b));
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    absl::StatusOr<xffi::Result<xffi::AnyBuffer>> b =
        rets.get<xffi::AnyBuffer>(i);
    if (!b.ok()) return b.status();
    if ((*b)->element_type() == xla::TOKEN) continue;
    out.push_back(describe(**b));
  }
  std::string error(4096, '\0');
  int rc = trampoline(callback_id, static_cast<int64_t>(in.size()), in.data(),
                      static_cast<int64_t>(out.size()), out.data(),
                      error.data(), static_cast<int64_t>(error.size()));
  if (rc != 0) {
    error.resize(error.find('\0'));
    return absl::InternalError(
        absl::StrCat("Metal host callback failed: ", error));
  }
  return absl::OkStatus();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalPythonCallback, PythonCallback,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Attr<uint64_t>("callback_id")
                           .RemainingArgs()
                           .RemainingRets());

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "xla_ffi_python_metal_callback",
                         "METAL", kMetalPythonCallback);

}  // namespace ffi
}  // namespace metal_pjrt

extern "C" void metal_pjrt_register_python_callback_trampoline(
    MetalPjrtPythonCallbackTrampoline fn) {
  metal_pjrt::ffi::g_trampoline.store(fn);
}
