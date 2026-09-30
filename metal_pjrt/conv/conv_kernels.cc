// Parts of this file are ported from MLX (mlx/backend/metal/conv.cpp, MLX
// 0.32.2), Copyright (c) 2023 Apple Inc., MIT License: see
// THIRD_PARTY_NOTICES.

#include "metal_pjrt/conv/conv_kernels.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "metal_pjrt/kernels/conv_misc.metal.h"
#include "metal_pjrt/kernels/steel_conv.metal.h"

namespace metal_pjrt {
namespace conv {

const char* MslTypeName(ConvType t) {
  switch (t) {
    case ConvType::kF32:
      return "float";
    case ConvType::kF16:
      return "half";
    case ConvType::kBF16:
      return "bfloat";
  }
  return "?";
}

ConvTile ImplicitTile(int64_t m, int64_t n, int64_t c) {
  ConvTile t;
  t.bm = m >= 8192 && c >= 64 ? 64 : 32;
  t.bn = (t.bm == 64 || n >= 64) ? 64 : 32;
  if (n <= 16) {
    t.bn = 8;
    t.wm = 4;
    t.wn = 1;
  }
  return t;
}

ConvTile GeneralTile(int64_t m, int64_t n, int64_t c) {
  ConvTile t;
  t.bm = m >= 8192 && c >= 64 ? 64 : 32;
  t.bn = (t.bm == 64 && n >= 64) ? 64 : 32;
  return t;
}

ConvKernelSource ImplicitConvKernel(ConvType t, const ConvTile& tile,
                                    int n_channels, bool small_filter) {
  const std::string name = absl::StrCat(
      "implicit_gemm_conv_2d_", MslTypeName(t), "_", tile.bm, "x", tile.bn,
      "x", tile.bk, "_", tile.wm, "x", tile.wn, "_c", n_channels, "_f",
      small_filter ? "s" : "l");
  return {absl::StrCat(kernels::kSteelConvMsl,
                       "\ninstantiate_implicit_gemm_conv_2d(\"", name, "\", ",
                       MslTypeName(t), ", ", tile.bm, ", ", tile.bn, ", ",
                       tile.bk, ", ", tile.wm, ", ", tile.wn, ", ",
                       n_channels, ", ", small_filter ? "true" : "false",
                       ")\n"),
          name,
          {}};
}

ConvKernelSource GeneralConvKernel(ConvType t, const ConvTile& tile,
                                   bool align_c) {
  const std::string name = absl::StrCat(
      "implicit_gemm_conv_2d_general_", MslTypeName(t), "_", tile.bm, "x",
      tile.bn, "x", tile.bk, "_", tile.wm, "x", tile.wn);
  return {absl::StrCat(kernels::kSteelConvMsl,
                       "\ninstantiate_implicit_gemm_conv_2d_general(\"", name,
                       "\", ", MslTypeName(t), ", ", tile.bm, ", ", tile.bn,
                       ", ", tile.bk, ", ", tile.wm, ", ", tile.wn, ")\n"),
          name,
          // The function_constant index of ALIGN_C in steel_conv.metal.
          {rt::FunctionConstant::Bool(0, align_c)}};
}

ConvKernelSource UnfoldKernel(int vector_bytes) {
  return {kernels::kConvMiscMsl, absl::StrCat("unfold_2d_", vector_bytes),
          {}};
}

ConvKernelSource PadColsKernel(ConvType t) {
  return {kernels::kConvMiscMsl, absl::StrCat("pad_cols_", MslTypeName(t)),
          {}};
}

ConvKernelSource SumSplitsKernel(ConvType t) {
  return {kernels::kConvMiscMsl, absl::StrCat("sum_splits_", MslTypeName(t)),
          {}};
}

std::vector<ConvKernelSource> AllConvKernels() {
  std::vector<ConvKernelSource> out;
  std::set<std::string> seen;
  auto add = [&](ConvKernelSource k) {
    std::string key = k.function;
    for (const rt::FunctionConstant& c : k.constants) {
      absl::StrAppend(&key, "/", c.index, "=", c.value);
    }
    if (seen.insert(key).second) out.push_back(std::move(k));
  };
  // The branch points of the tile rules: m 8192, c 64 (and the channel
  // specializations 1-4), n 16 and 64.
  for (ConvType t : {ConvType::kF32, ConvType::kF16, ConvType::kBF16}) {
    for (int64_t m : {1, 8192}) {
      for (int64_t n : {1, 32, 64}) {
        for (int64_t c : {1, 2, 3, 4, 16, 64}) {
          const ConvTile it = ImplicitTile(m, n, c);
          if (c <= 4) {
            add(ImplicitConvKernel(t, it, static_cast<int>(c), false));
          } else {
            for (bool small : {false, true}) {
              add(ImplicitConvKernel(t, it, 0, small));
            }
          }
          for (bool align : {false, true}) {
            add(GeneralConvKernel(t, GeneralTile(m, n, c), align));
          }
        }
      }
    }
    add(PadColsKernel(t));
    add(SumSplitsKernel(t));
  }
  for (int bytes : {2, 4, 8, 16}) add(UnfoldKernel(bytes));
  return out;
}

}  // namespace conv
}  // namespace metal_pjrt
