// "metal$test_scale": y = x * scale for f32 buffers. Test-only (not linked
// into the plugin); ffi_test uses it to exercise handler registration and
// LaunchMsl.
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "metal_pjrt_plugin/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

constexpr char kTestScaleMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; float scale; };
kernel void test_scale(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       constant Params& p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < p.n) y[i] = x[i] * p.scale;
}
)msl";

struct Params {
  uint32_t n;
  float scale;
};

absl::Status TestScale(stream_executor::Stream* stream, xffi::AnyBuffer x,
                       xffi::Result<xffi::AnyBuffer> y, float scale) {
  if (x.element_type() != xla::F32 || y->element_type() != xla::F32) {
    return absl::InvalidArgumentError("metal$test_scale: f32 only");
  }
  if (x.element_count() != y->element_count()) {
    return absl::InvalidArgumentError("metal$test_scale: size mismatch");
  }
  Params p{static_cast<uint32_t>(x.element_count()), scale};
  if (p.n == 0) return absl::OkStatus();
  constexpr uint32_t kThreads = 256;
  return LaunchMsl(stream, kTestScaleMsl, "test_scale",
                   {x.untyped_data(), y->untyped_data()}, p,
                   rt::Dim3{(p.n + kThreads - 1) / kThreads, 1, 1},
                   rt::Dim3{kThreads, 1, 1});
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalTestScale, TestScale,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<float>("scale"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$test_scale", "METAL",
                         kMetalTestScale);

}  // namespace ffi
}  // namespace metal_pjrt
