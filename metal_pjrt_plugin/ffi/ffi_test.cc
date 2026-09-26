// FFI plumbing: a handler registered statically for platform "METAL" is found
// in XLA's registry and runs on a Metal stream through LaunchMsl.
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace se = ::stream_executor;
namespace xffi = ::xla::ffi;

TEST(FfiTest, RegisteredHandlerRunsOnMetalStream) {
  TF_ASSERT_OK_AND_ASSIGN(xffi::HandlerRegistration handler,
                          xffi::FindHandler("metal$test_scale", "METAL"));
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  const int64_t n = 1000;
  se::DeviceAddressBase x = executor->Allocate(n * sizeof(float));
  se::DeviceAddressBase y = executor->Allocate(n * sizeof(float));
  ASSERT_FALSE(x.is_null());
  ASSERT_FALSE(y.is_null());
  std::vector<float> hx(n), out(n, 0.0f);
  for (int64_t i = 0; i < n; ++i) hx[i] = static_cast<float>(i);
  TF_ASSERT_OK(stream->Memcpy(&x, hx.data(), n * sizeof(float)));

  xffi::CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/1);
  builder.AddBufferArg(x, xla::F32, {n});
  builder.AddBufferRet(y, xla::F32, {n});
  xffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("scale", xffi::Attribute{xffi::Scalar{2.5f}});
  builder.AddAttributes(attrs.Build());
  xffi::CallFrame frame = builder.Build();

  xffi::InvokeContext context;
  context.backend_context = xffi::InvokeContext::GpuContext{stream.get()};
  TF_ASSERT_OK(xffi::Invoke(xffi::GetXlaFfiApi(), handler.bundle.execute,
                            frame, context));
  TF_ASSERT_OK(stream->Memcpy(out.data(), y, n * sizeof(float)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  for (int64_t i = 0; i < n; ++i) ASSERT_EQ(out[i], 2.5f * i) << i;

  executor->Deallocate(&x);
  executor->Deallocate(&y);
}

}  // namespace
}  // namespace ffi
}  // namespace metal_pjrt
