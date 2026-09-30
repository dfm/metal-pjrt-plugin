// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "metal_pjrt/compiler/metal_compiler.h"
#include "metal_pjrt/stream_executor/metal_platform_id.h"
#include "xla/service/compiler.h"

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::metal::kMetalPlatformId,
      []() { return std::make_unique<xla::gpu::MetalCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
