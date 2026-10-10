// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/stream_executor/metal_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "metal_pjrt/stream_executor/metal_executor.h"
#include "metal_pjrt/stream_executor/metal_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor {
namespace metal {

MetalPlatform::MetalPlatform() : name_(kMetalPlatformId->ToName()) {}

MetalPlatform::~MetalPlatform() = default;

PlatformId MetalPlatform::id() const { return kMetalPlatformId; }

const std::string& MetalPlatform::Name() const { return name_; }

int MetalPlatform::VisibleDeviceCount() const {
  return metal_pjrt::rt::Device::VisibleDeviceCount();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalPlatform::DescriptionForDevice(int ordinal) const {
  return MetalExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> MetalPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> MetalPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
MetalPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<MetalExecutor>(this, ordinal);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing Metal StreamExecutor for device ordinal %d: %s",
        ordinal, init_status.ToString()));
  }
  return std::move(executor);
}

static void InitializeMetalPlatform() {
  if (PlatformManager::PlatformWithName("METAL").ok()) return;
  std::unique_ptr<Platform> platform(new MetalPlatform);
  CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace metal
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    metal_platform, stream_executor::metal::InitializeMetalPlatform());
