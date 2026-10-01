// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// "metal$pool_max_bwd": XLA's select-and-scatter for a max pool's gradient
// over non-overlapping windows (compiler/passes/pool_rewriter.h); the kernel
// and its dispatch are in pool.h.
//
// Operands: x [d...] and dy [d / window...] (row-major, one element type,
// f32/f16/bf16); result dx, x's shape and type. Attributes: window (i64
// array, x's rank), init (f32, the scatter's initial value).
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/ffi/pool.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

absl::Status PoolMaxBwd(stream_executor::Stream* stream, xffi::AnyBuffer x,
                        xffi::AnyBuffer dy, xffi::Result<xffi::AnyBuffer> dx,
                        absl::Span<const int64_t> window, float init) {
  PoolType type;
  switch (x.element_type()) {
    case xla::F32:
      type = PoolType::kF32;
      break;
    case xla::F16:
      type = PoolType::kF16;
      break;
    case xla::BF16:
      type = PoolType::kBF16;
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "metal$pool_max_bwd: unsupported element type ",
          xla::primitive_util::LowercasePrimitiveTypeName(x.element_type())));
  }
  auto dims = x.dimensions();
  auto dy_dims = dy.dimensions();
  bool ok = dy.element_type() == x.element_type() &&
            dx->element_type() == x.element_type() &&
            dx->element_count() == x.element_count() &&
            dy_dims.size() == dims.size() && window.size() == dims.size();
  for (size_t i = 0; ok && i < dims.size(); ++i) {
    ok = window[i] >= 1 && dy_dims[i] == dims[i] / window[i];
  }
  if (!ok) {
    return absl::InvalidArgumentError(
        "metal$pool_max_bwd: operand/result shapes do not match the window");
  }
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  return RunPoolMaxBwd(ctx->device, ctx->stream, x.untyped_data(),
                       dy.untyped_data(), dx->untyped_data(), dims, window,
                       init, type);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalPoolMaxBwd, PoolMaxBwd,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::Span<const int64_t>>("window")
                           .Attr<float>("init"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$pool_max_bwd", "METAL",
                         kMetalPoolMaxBwd);

}  // namespace ffi
}  // namespace metal_pjrt
