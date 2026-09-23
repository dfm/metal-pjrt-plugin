#include <memory>

#include "absl/log/check.h"
#include "metal_pjrt_plugin/pjrt/metal_platform_name.h"
#include "metal_pjrt_plugin/stream_executor/metal_platform_id.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/se/stream_executor_platform_id_mapping.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla {

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    pjrt_register_se_gpu_metal_compiler, {
      PjRtRegisterDefaultCompiler(
          MetalName(), std::make_unique<StreamExecutorGpuCompiler>(
                           MetalId(), stream_executor::metal::kMetalPlatformId));
      CHECK_OK(StreamExecutorPlatformIdMapping::Global().AddMapping(
          stream_executor::metal::kMetalPlatformId, MetalId()));
    });

}  // namespace xla
