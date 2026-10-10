// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// "metal$scan": inclusive scan (cumsum/cumprod/cummax/cummin) over the minor
// dimension; the kernel and its dispatch are in scan.h.
//
// Operand and result: same shape [..., n], row-major, f32/f16/bf16/s32.
// Attributes: op ("add" | "mul" | "max" | "min"), reverse (bool),
// row_length (i64, must equal the minor dim).
//
// The kernel is looked up once per call site, at the FFI instantiate stage
// (ScanState); execution only encodes the dispatch.
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/ffi/scan.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

absl::StatusOr<ScanType> GetScanType(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32:
      return ScanType::kF32;
    case xla::F16:
      return ScanType::kF16;
    case xla::BF16:
      return ScanType::kBF16;
    case xla::S32:
      return ScanType::kS32;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Metal FFI: unsupported element type ",
                       xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

struct ScanState {
  ScanType type;
  ScanOp op;
  // Compiled at instantiation for DefaultMetalDevice().
  rt::Device* device = nullptr;
  const rt::Kernel* kernel = nullptr;
};

absl::Status CheckShapes(const xffi::AnyBuffer& x, const xffi::AnyBuffer& y,
                         int64_t row_length) {
  auto dims = x.dimensions();
  if (dims.empty() || dims.back() != row_length ||
      x.element_type() != y.element_type() ||
      x.element_count() != y.element_count()) {
    return absl::InvalidArgumentError(
        "metal$scan: operand/result shape mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ScanState>> Instantiate(
    xffi::AnyBuffer x, xffi::Result<xffi::AnyBuffer> y, absl::string_view op,
    bool reverse, int64_t row_length) {
  if (absl::Status s = CheckShapes(x, *y, row_length); !s.ok()) return s;
  absl::StatusOr<ScanType> type = GetScanType(x.element_type());
  if (!type.ok()) return type.status();
  absl::StatusOr<ScanOp> scan_op = ParseScanOp(op);
  if (!scan_op.ok()) return scan_op.status();
  auto state = std::make_unique<ScanState>();
  state->type = *type;
  state->op = *scan_op;
  absl::StatusOr<rt::Device*> device = DefaultMetalDevice();
  if (!device.ok()) return device.status();
  absl::StatusOr<const rt::Kernel*> kernel =
      GetScanKernel(*device, state->type, state->op);
  if (!kernel.ok()) return kernel.status();
  state->device = *device;
  state->kernel = *kernel;
  return state;
}

absl::Status Scan(stream_executor::Stream* stream, xffi::AnyBuffer x,
                  xffi::Result<xffi::AnyBuffer> y, absl::string_view op,
                  bool reverse, int64_t row_length, ScanState* state) {
  if (absl::Status s = CheckShapes(x, *y, row_length); !s.ok()) return s;
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  const rt::Kernel* kernel = state->kernel;
  if (ctx->device != state->device) {
    absl::StatusOr<const rt::Kernel*> k =
        GetScanKernel(ctx->device, state->type, state->op);
    if (!k.ok()) return k.status();
    kernel = *k;
  }
  const uint64_t n = static_cast<uint64_t>(row_length);
  const uint64_t rows = n == 0 ? 0 : x.element_count() / n;
  return RunScan(ctx->device, ctx->stream, x.untyped_data(), y->untyped_data(),
                 rows, n, state->type, state->op, reverse, kernel);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalScanInstantiate, Instantiate,
                       xffi::Ffi::BindInstantiate()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::string_view>("op")
                           .Attr<bool>("reverse")
                           .Attr<int64_t>("row_length"));

XLA_FFI_DEFINE_HANDLER(kMetalScan, Scan,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::string_view>("op")
                           .Attr<bool>("reverse")
                           .Attr<int64_t>("row_length")
                           .Ctx<xffi::State<ScanState>>());

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$scan", "METAL",
                         {/*instantiate=*/kMetalScanInstantiate,
                          /*prepare=*/nullptr, /*initialize=*/nullptr,
                          /*execute=*/kMetalScan});

}  // namespace ffi
}  // namespace metal_pjrt
