// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// "metal$conv": a 2-D convolution or weight gradient on MLX's steel kernels
// (metal_pjrt/conv/conv.h), the target of MetalConvRewriter
// (compiler/passes/conv_rewriter.h).
//
// Operands: input [N, H, W, C] and, for kind "fwd", the weight [O, kH, kW,
// C] or, for kind "wgrad", the output gradient [N, oH, oW, O]. Results: the
// output [N, oH, oW, O] ("fwd") or the weight gradient [O, kH, kW, C]
// ("wgrad"), then a u8 workspace of at least PlanConv's workspace_bytes.
// All row-major, one type (f32/f16/bf16). Attributes: kind, stride, pad_lo,
// kdil, idil (array<i64: h, w>), flip (bool); the high padding follows
// from the sizes.
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt/conv/conv.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

absl::StatusOr<conv::ConvType> GetConvType(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32:
      return conv::ConvType::kF32;
    case xla::F16:
      return conv::ConvType::kF16;
    case xla::BF16:
      return conv::ConvType::kBF16;
    default:
      return absl::UnimplementedError(
          absl::StrCat("metal$conv: unsupported element type ",
                       xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

absl::StatusOr<conv::ConvParams> Params(const xffi::AnyBuffer& in,
                                        const xffi::AnyBuffer& b,
                                        const xffi::AnyBuffer& out,
                                        absl::string_view kind,
                                        absl::Span<const int64_t> stride,
                                        absl::Span<const int64_t> pad_lo,
                                        absl::Span<const int64_t> kdil,
                                        absl::Span<const int64_t> idil,
                                        bool flip) {
  auto x = in.dimensions(), y = b.dimensions(), z = out.dimensions();
  if (x.size() != 4 || y.size() != 4 || z.size() != 4 ||
      in.element_type() != b.element_type() ||
      in.element_type() != out.element_type() || stride.size() != 2 ||
      pad_lo.size() != 2 || kdil.size() != 2 || idil.size() != 2) {
    return absl::InvalidArgumentError("metal$conv: bad operands/attributes");
  }
  conv::ConvParams p;
  ABSL_ASSIGN_OR_RETURN(p.type, GetConvType(in.element_type()));
  p.n = x[0];
  p.h = x[1];
  p.w = x[2];
  p.c = x[3];
  if (kind == "fwd") {
    p.kind = conv::ConvKind::kForward;
    p.o = y[0];
    p.kh = y[1];
    p.kw = y[2];
    p.out_h = z[1];
    p.out_w = z[2];
    if (y[3] != p.c || z[0] != p.n || z[3] != p.o) {
      return absl::InvalidArgumentError("metal$conv: shape mismatch (fwd)");
    }
  } else if (kind == "wgrad") {
    p.kind = conv::ConvKind::kWeightGrad;
    p.o = z[0];
    p.kh = z[1];
    p.kw = z[2];
    p.out_h = y[1];
    p.out_w = y[2];
    if (z[3] != p.c || y[0] != p.n || y[3] != p.o) {
      return absl::InvalidArgumentError("metal$conv: shape mismatch (wgrad)");
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("metal$conv: unknown kind ", kind));
  }
  for (int i = 0; i < 2; ++i) {
    p.stride[i] = stride[i];
    p.pad_lo[i] = pad_lo[i];
    p.kdil[i] = kdil[i];
    p.idil[i] = idil[i];
  }
  p.flip = flip;
  return p;
}

absl::Status Conv(stream_executor::Stream* stream, xffi::AnyBuffer in,
                  xffi::AnyBuffer b, xffi::Result<xffi::AnyBuffer> out,
                  xffi::Result<xffi::BufferR1<xla::U8>> workspace,
                  absl::string_view kind, absl::Span<const int64_t> stride,
                  absl::Span<const int64_t> pad_lo,
                  absl::Span<const int64_t> kdil,
                  absl::Span<const int64_t> idil, bool flip) {
  ABSL_ASSIGN_OR_RETURN(
      conv::ConvParams p,
      Params(in, b, *out, kind, stride, pad_lo, kdil, idil, flip));
  ABSL_ASSIGN_OR_RETURN(conv::ConvPlan plan, conv::PlanConv(p));
  if (workspace->size_bytes() < plan.workspace_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metal$conv: workspace of ", workspace->size_bytes(),
        " bytes, the plan needs ", plan.workspace_bytes));
  }
  ABSL_ASSIGN_OR_RETURN(MetalContext ctx, GetMetalContext(stream));
  return conv::RunConv(
      ctx.device, ctx.stream, p, plan, in.untyped_data(), b.untyped_data(),
      out->untyped_data(),
      plan.workspace_bytes > 0 ? workspace->untyped_data() : nullptr);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalConv, Conv,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::BufferR1<xla::U8>>()  // workspace
                           .Attr<absl::string_view>("kind")
                           .Attr<absl::Span<const int64_t>>("stride")
                           .Attr<absl::Span<const int64_t>>("pad_lo")
                           .Attr<absl::Span<const int64_t>>("kdil")
                           .Attr<absl::Span<const int64_t>>("idil")
                           .Attr<bool>("flip"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$conv", "METAL",
                         kMetalConv);

}  // namespace ffi
}  // namespace metal_pjrt
