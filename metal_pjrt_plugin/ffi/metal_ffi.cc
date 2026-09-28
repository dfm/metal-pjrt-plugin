#include "metal_pjrt_plugin/ffi/metal_ffi.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"
#include "metal_pjrt_plugin/stream_executor/metal_platform_id.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace metal_pjrt {
namespace ffi {

absl::StatusOr<MetalContext> GetMetalContext(stream_executor::Stream* stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError("Metal FFI handler: null stream");
  }
  stream_executor::StreamExecutor* executor = stream->parent();
  if (executor == nullptr || executor->GetPlatform() == nullptr ||
      executor->GetPlatform()->id() !=
          stream_executor::metal::kMetalPlatformId) {
    return absl::InvalidArgumentError(
        "Metal FFI handler invoked on a stream of another platform");
  }
  auto* metal_executor =
      static_cast<stream_executor::metal::MetalExecutor*>(executor);
  MetalContext ctx;
  ctx.stream =
      static_cast<rt::Stream*>(stream->platform_specific_handle().stream);
  ctx.device = metal_executor->device();
  if (ctx.stream == nullptr || ctx.device == nullptr) {
    return absl::InternalError("Metal FFI handler: stream has no runtime");
  }
  return ctx;
}

absl::StatusOr<rt::Device*> DefaultMetalDevice() {
  absl::StatusOr<stream_executor::Platform*> platform =
      stream_executor::PlatformManager::PlatformWithId(
          stream_executor::metal::kMetalPlatformId);
  if (!platform.ok()) return platform.status();
  absl::StatusOr<stream_executor::StreamExecutor*> executor =
      (*platform)->ExecutorForDevice(0);
  if (!executor.ok()) return executor.status();
  rt::Device* device =
      static_cast<stream_executor::metal::MetalExecutor*>(*executor)->device();
  if (device == nullptr) {
    return absl::InternalError("Metal FFI: executor 0 has no runtime device");
  }
  return device;
}

absl::StatusOr<std::string> MslTypeName(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32:
      return std::string("float");
    case xla::F16:
      return std::string("half");
    case xla::BF16:
      return std::string("bfloat");
    case xla::S32:
      return std::string("int");
    default:
      return absl::UnimplementedError(
          absl::StrCat("Metal FFI: unsupported element type ",
                       xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

}  // namespace ffi
}  // namespace metal_pjrt
