// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/ffi/metal_ffi.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/stream_executor/metal_executor.h"
#include "metal_pjrt/stream_executor/metal_platform_id.h"
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

}  // namespace ffi
}  // namespace metal_pjrt
