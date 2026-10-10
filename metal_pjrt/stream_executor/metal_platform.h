// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_H_
#define METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/executor_cache.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace metal {

// Apple Metal platform: one executor per MTLDevice.
class MetalPlatform : public Platform {
 public:
  MetalPlatform();
  ~MetalPlatform() override;

  PlatformId id() const override;
  const std::string& Name() const override;
  int VisibleDeviceCount() const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  absl::StatusOr<StreamExecutor*> FindExisting(int ordinal) override;

 private:
  absl::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      int ordinal);

  std::string name_;
  ExecutorCache executor_cache_;

  MetalPlatform(const MetalPlatform&) = delete;
  void operator=(const MetalPlatform&) = delete;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_H_
