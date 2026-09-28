#include "metal_pjrt_plugin/blas/blas_lt_support.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace metal {
namespace {

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

}  // namespace

absl::StatusOr<ValidatedMatmul> ValidateMatmul(const gpu::GemmConfig& cfg,
                                               gpu::BlasLt::Epilogue epilogue) {
  ValidatedMatmul v{/*swap=*/false, cfg.lhs_layout, cfg.rhs_layout,
                    cfg.c_layout, cfg.output_layout};
  ABSL_ASSIGN_OR_RETURN(v.epi, DecodeEpilogue(epilogue));
  if (cfg.alpha.imag() != 0.0) {
    return absl::UnimplementedError("Metal BlasLt: complex alpha");
  }
  for (const gpu::MatrixLayout* l : {&v.lhs, &v.rhs, &v.c, &v.out}) {
    if (l->transpose != blas::Transpose::kNoTranspose) {
      return absl::UnimplementedError(
          absl::StrCat("Metal BlasLt: transposed layout ", l->ToString()));
    }
  }
  // As in cuBLASLt: an operand without a batch dimension is broadcast
  // (its batch_stride is 0).
  v.batch = std::max(v.lhs.batch_size, v.rhs.batch_size);
  v.lhs.batch_size = v.rhs.batch_size = v.batch;
  if (v.out.batch_size != v.batch) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal BlasLt: batch mismatch ", v.out.ToString()));
  }

  // MPS and steel write a row-major result. For a column-major output use
  //   D^T = (A B)^T = B^T A^T,
  // i.e. swap the operands and view every layout transposed (the inverse of
  // gpu::MakeOutputColumnMajor).
  v.swap = v.out.order == gpu::MatrixLayout::Order::kColumnMajor;
  if (v.swap) {
    std::swap(v.lhs, v.rhs);
    v.lhs.Transpose();
    v.rhs.Transpose();
    v.c.Transpose();
    v.out.Transpose();
  }
  v.m = v.out.num_rows;
  v.n = v.out.num_cols;
  v.k = v.lhs.num_cols;
  if (v.lhs.num_rows != v.m || v.rhs.num_rows != v.k ||
      v.rhs.num_cols != v.n) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal BlasLt: inconsistent shapes lhs=", v.lhs.ToString(),
        " rhs=", v.rhs.ToString(), " out=", v.out.ToString()));
  }
  ABSL_RETURN_IF_ERROR(
      CheckBlasLtTypes(v.lhs.dtype, v.rhs.dtype, v.out.dtype));
  const gpu::MatrixLayout& c = v.c;
  const gpu::MatrixLayout& out = v.out;
  if (cfg.beta != 0.0 &&
      (c.dtype != out.dtype || c.order != out.order ||
       c.leading_dim_stride != out.leading_dim_stride ||
       c.batch_stride != out.batch_stride || c.num_rows != out.num_rows ||
       c.num_cols != out.num_cols)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal BlasLt: C layout ", c.ToString(),
                     " differs from output layout ", out.ToString()));
  }

  const xla::PrimitiveType t = v.lhs.dtype;
  if (t == xla::F16 || t == xla::BF16) {
    // Steel's index math is 32-bit: every dimension and the batch count must
    // fit in int32, and every leading dimension in kMaxLd elements (ld * 128,
    // the largest tile extent, stays in range; SteelGemmSupports in
    // steel_gemm.cc, which is XLA-free, checks the same before encoding).
    constexpr int64_t kMaxInt = 2147483647;
    constexpr int64_t kMaxLd = kMaxInt / 256;
    auto refuse = [&](absl::string_view what, int64_t value, int64_t limit) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal BlasLt: ", xla::PrimitiveType_Name(t),
          " GEMMs run on the steel kernels, whose index math is 32-bit: ",
          what, " ", value, " exceeds ", limit));
    };
    if (v.batch > kMaxInt) return refuse("batch count", v.batch, kMaxInt);
    for (int64_t d : {v.m, v.n, v.k}) {
      if (d > kMaxInt) return refuse("dimension", d, kMaxInt);
    }
    for (const gpu::MatrixLayout* l : {&v.lhs, &v.rhs, &v.out}) {
      if (l->leading_dim_stride > kMaxLd) {
        return refuse("leading dimension (elements between rows)",
                      l->leading_dim_stride, kMaxLd);
      }
    }
  } else if (!v.epi.trivial() &&
             (v.m > UINT32_MAX || v.n > UINT32_MAX || v.batch > UINT32_MAX)) {
    // f32: MPS, then an epilogue kernel over the row-major view of D whose
    // grid is (n, m, batch); Metal bounds each by uint32.
    return absl::UnimplementedError(
        absl::StrCat("Metal BlasLt: epilogue on a ", v.batch, "x", v.m, "x",
                     v.n, " output"));
  }
  return v;
}

}  // namespace metal
}  // namespace stream_executor
