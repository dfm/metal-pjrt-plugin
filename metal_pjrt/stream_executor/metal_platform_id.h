// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_ID_H_
#define METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_ID_H_

#include "xla/stream_executor/platform_id.h"

namespace stream_executor {
namespace metal {

// Opaque identifier for the Apple Metal platform. Name() is "METAL".
extern const PlatformId kMetalPlatformId;

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_STREAM_EXECUTOR_METAL_PLATFORM_ID_H_
