#include "metal_pjrt_plugin/blas/blas_lt_support.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace metal {

absl::StatusOr<EpilogueSpec> DecodeEpilogue(gpu::BlasLt::Epilogue e) {
  using E = gpu::BlasLt::Epilogue;
  EpilogueSpec s;
  switch (e) {
    case E::kDefault:
      break;
    case E::kReLU:
      s.act = Activation::kReLU;
      break;
    case E::kBias:
      s.bias = true;
      break;
    case E::kBiasThenReLU:
      s.bias = true;
      s.act = Activation::kReLU;
      break;
    case E::kGELU:
      s.act = Activation::kGELU;
      break;
    case E::kGELUWithAux:
      s.act = Activation::kGELU;
      s.aux = true;
      break;
    case E::kBiasThenGELU:
      s.bias = true;
      s.act = Activation::kGELU;
      break;
    case E::kBiasThenGELUWithAux:
      s.bias = true;
      s.act = Activation::kGELU;
      s.aux = true;
      break;
    case E::kSILU:
      s.act = Activation::kSILU;
      break;
    case E::kSILUWithAux:
      s.act = Activation::kSILU;
      s.aux = true;
      break;
    case E::kBiasThenSILU:
      s.bias = true;
      s.act = Activation::kSILU;
      break;
    case E::kBiasThenSILUWithAux:
      s.bias = true;
      s.act = Activation::kSILU;
      s.aux = true;
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Metal BlasLt: epilogue ", static_cast<int>(e),
                       " is not supported"));
  }
  return s;
}


absl::Status CheckBlasLtTypes(xla::PrimitiveType a, xla::PrimitiveType b,
                              xla::PrimitiveType out) {
  auto ok = [](xla::PrimitiveType t) {
    return t == xla::F32 || t == xla::F16 || t == xla::BF16;
  };
  if (ok(a) && a == b && ok(out) && (out == a || out == xla::F32)) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(absl::StrCat(
      "Metal BlasLt: unsupported types ", xla::PrimitiveType_Name(a), " x ",
      xla::PrimitiveType_Name(b), " -> ", xla::PrimitiveType_Name(out)));
}

absl::Status CheckBlasLtShape(const gpu::GemmConfig& cfg) {
  const xla::PrimitiveType t = cfg.lhs_layout.dtype;
  if (t != xla::F16 && t != xla::BF16) return absl::OkStatus();
  constexpr int64_t kMaxInt = 2147483647;
  auto refuse = [&](absl::string_view what, int64_t value, int64_t limit) {
    return absl::UnimplementedError(absl::StrCat(
        "Metal BlasLt: ", xla::PrimitiveType_Name(t),
        " GEMMs run on the steel kernels, whose index math is 32-bit: ", what,
        " ", value, " exceeds ", limit));
  };
  const int64_t batch =
      std::max(cfg.lhs_layout.batch_size, cfg.rhs_layout.batch_size);
  if (batch > kMaxInt) return refuse("batch count", batch, kMaxInt);
  for (const gpu::MatrixLayout* l :
       {&cfg.lhs_layout, &cfg.rhs_layout, &cfg.output_layout}) {
    if (l->num_rows > kMaxInt) return refuse("dimension", l->num_rows, kMaxInt);
    if (l->num_cols > kMaxInt) return refuse("dimension", l->num_cols, kMaxInt);
    if (l->leading_dim_stride > kSteelMaxLd) {
      return refuse("leading dimension (elements between rows)",
                    l->leading_dim_stride, kSteelMaxLd);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace stream_executor
