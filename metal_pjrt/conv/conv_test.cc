// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// RunConv (conv.h) against a double reference on rt::Device alone (no XLA):
// for every case, every path PlanConv accepts (the one it picks, and each
// forced one), in f32, f16 and bf16. Cases: the cnn bench's layers and its
// input gradient (input dilation + flip), channel counts around the kernels'
// specializations (C 1-5, 16, 32; O 8, 16, 17, 64), strides, kernel
// dilation, asymmetric padding, a kernel wider than the small-filter masks
// (with flip), several unfold tiles on the explicit path, and the path
// choice itself. f16/bf16 accumulate in f32, so all three types are within
// the f32 accumulation error of the reference on the rounded inputs, plus
// the output rounding. The weight gradient (kWeightGrad) likewise, over its
// chunks and split-K parts; and every path split over several launches.
// The unfold matches the naive kernel it replaced byte for byte. Needs a
// Metal device.
#include "metal_pjrt/conv/conv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace conv {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

constexpr ConvType kTypes[] = {ConvType::kF32, ConvType::kF16,
                               ConvType::kBF16};
constexpr ConvPath kPaths[] = {ConvPath::kImplicit, ConvPath::kPadChannels,
                               ConvPath::kGeneral, ConvPath::kExplicit};

int Bytes(ConvType t) { return t == ConvType::kF32 ? 4 : 2; }

float Bf16ToFloat(uint16_t b) {
  uint32_t u = uint32_t{b} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t FloatToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}

double Load(const std::vector<uint8_t>& buf, int64_t i, ConvType t) {
  switch (t) {
    case ConvType::kF32: {
      float f;
      std::memcpy(&f, buf.data() + i * 4, 4);
      return f;
    }
    case ConvType::kF16: {
      _Float16 h;
      std::memcpy(&h, buf.data() + i * 2, 2);
      return static_cast<float>(h);
    }
    case ConvType::kBF16: {
      uint16_t b;
      std::memcpy(&b, buf.data() + i * 2, 2);
      return Bf16ToFloat(b);
    }
  }
  return 0;
}

void Store(std::vector<uint8_t>& buf, int64_t i, ConvType t, double v) {
  switch (t) {
    case ConvType::kF32: {
      float f = static_cast<float>(v);
      std::memcpy(buf.data() + i * 4, &f, 4);
      break;
    }
    case ConvType::kF16: {
      _Float16 h = static_cast<_Float16>(static_cast<float>(v));
      std::memcpy(buf.data() + i * 2, &h, 2);
      break;
    }
    case ConvType::kBF16: {
      uint16_t b = FloatToBf16(static_cast<float>(v));
      std::memcpy(buf.data() + i * 2, &b, 2);
      break;
    }
  }
}

// The unfold before the vectorized one (MLX's naive_unfold_Nd for 2-D, one
// element per thread), the reference Unfold must match byte for byte. A
// bit copy: T is ushort or uint.
constexpr char kNaiveUnfoldMsl[] = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct SteelConvParams {
  int N; int C; int O; int iS[2]; int wS[2]; int oS[2]; int str[2];
  int pad[2]; int kdil[2]; int idil[2]; long in_strides[4];
  long wt_strides[4]; long out_strides[4]; int groups; bool flip;
};
struct UnfoldRows { int row_offset; int rows; int vecs; };
template <typename T>
[[kernel]] void naive_unfold_2d(const device T* in [[buffer(0)]],
                                device T* out [[buffer(1)]],
                                const constant SteelConvParams* params
                                [[buffer(2)]],
                                const constant UnfoldRows* tile [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
  const int filter_size = params->C * params->wS[0] * params->wS[1];
  if (int(gid.x) >= filter_size || int(gid.y) >= tile->rows) return;
  const int c = int(gid.x) % params->C;
  const int out_pixels = params->oS[0] * params->oS[1];
  out += size_t(gid.y) * filter_size + (gid.x - c);
  int is[2] = {0, 0};
  const int global_row = tile->row_offset + int(gid.y);
  const int n = global_row / out_pixels;
  int oS = global_row % out_pixels;
  int wS = int(gid.x) / params->C;
  bool valid = n < params->N;
  for (int i = 1; i >= 0; --i) {
    const int os_ = oS % params->oS[i];
    int ws_ = wS % params->wS[i];
    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;
    const int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    const int is_max = 1 + params->idil[i] * (params->iS[i] - 1);
    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);
    is[i] = is_ / params->idil[i];
    oS /= params->oS[i];
    wS /= params->wS[i];
  }
  if (valid) {
    const size_t in_offset = size_t(n) * params->in_strides[0] +
                             size_t(is[0]) * params->in_strides[1] +
                             size_t(is[1]) * params->in_strides[2];
    out[c] = in[in_offset + c];
  } else {
    out[c] = T(0);
  }
}
template [[host_name("naive_unfold_2d_2")]] [[kernel]] decltype(
    naive_unfold_2d<ushort>) naive_unfold_2d<ushort>;
template [[host_name("naive_unfold_2d_4")]] [[kernel]] decltype(
    naive_unfold_2d<uint>) naive_unfold_2d<uint>;
)MSL";

// A case in terms of both paddings; the output size follows.
struct Case {
  int64_t n, h, w, c, o, kh, kw;
  int64_t stride[2] = {1, 1};
  int64_t pad_lo[2] = {0, 0};
  int64_t pad_hi[2] = {0, 0};
  int64_t kdil[2] = {1, 1};
  int64_t idil[2] = {1, 1};
  bool flip = false;
};

ConvParams Params(const Case& k, ConvType t) {
  ConvParams p;
  p.type = t;
  p.n = k.n;
  p.h = k.h;
  p.w = k.w;
  p.c = k.c;
  p.o = k.o;
  p.kh = k.kh;
  p.kw = k.kw;
  const int64_t in[2] = {k.h, k.w}, ks[2] = {k.kh, k.kw};
  int64_t out[2];
  for (int i = 0; i < 2; ++i) {
    p.stride[i] = k.stride[i];
    p.pad_lo[i] = k.pad_lo[i];
    p.kdil[i] = k.kdil[i];
    p.idil[i] = k.idil[i];
    const int64_t in_eff = (in[i] - 1) * k.idil[i] + 1;
    const int64_t k_eff = (ks[i] - 1) * k.kdil[i] + 1;
    out[i] = (in_eff + k.pad_lo[i] + k.pad_hi[i] - k_eff) / k.stride[i] + 1;
  }
  p.out_h = out[0];
  p.out_w = out[1];
  p.flip = k.flip;
  return p;
}

class ConvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }

  std::vector<uint8_t> Random(ConvType t, int64_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<uint8_t> x(count * Bytes(t));
    for (int64_t i = 0; i < count; ++i) Store(x, i, t, u(rng));
    return x;
  }

  // Runs `p` with `plan` and compares with the reference.
  void Check(const ConvParams& p, const ConvPlan& plan) {
    SCOPED_TRACE(absl::StrCat(ConvPathName(plan.path), " ",
                              ConvParamsDebugString(p)));
    const ConvType t = p.type;
    const int64_t n_in = p.n * p.h * p.w * p.c;
    const int64_t n_wt = p.o * p.kh * p.kw * p.c;
    const int64_t n_out = p.n * p.out_h * p.out_w * p.o;
    std::vector<uint8_t> x = Random(t, n_in, 17 + n_in);
    std::vector<uint8_t> wt = Random(t, n_wt, 29 + n_wt);
    absl::StatusOr<rt::Allocation> in = dev_->Allocate(x.size());
    absl::StatusOr<rt::Allocation> w = dev_->Allocate(wt.size());
    absl::StatusOr<rt::Allocation> out = dev_->Allocate(n_out * Bytes(t));
    ASSERT_THAT(in, IsOk());
    ASSERT_THAT(w, IsOk());
    ASSERT_THAT(out, IsOk());
    void* ws = nullptr;
    if (plan.workspace_bytes > 0) {
      absl::StatusOr<rt::Allocation> a = dev_->Allocate(plan.workspace_bytes);
      ASSERT_THAT(a, IsOk());
      ws = a->ptr;
      std::memset(ws, 0x7f, plan.workspace_bytes);  // stale values show
    }
    std::memcpy(in->ptr, x.data(), x.size());
    std::memcpy(w->ptr, wt.data(), wt.size());
    std::memset(out->ptr, 0xff, n_out * Bytes(t));  // missing writes show
    ASSERT_THAT(RunConv(dev_.get(), stream_.get(), p, plan, in->ptr, w->ptr,
                        out->ptr, ws),
                IsOk());
    ASSERT_THAT(stream_->Synchronize(), IsOk());
    std::vector<uint8_t> y(n_out * Bytes(t));
    std::memcpy(y.data(), out->ptr, y.size());
    for (void* ptr : {in->ptr, w->ptr, out->ptr, ws}) {
      if (ptr != nullptr) ASSERT_THAT(dev_->Deallocate(ptr), IsOk());
    }

    const double u32 = std::ldexp(1.0, -24);
    const double out_eps = t == ConvType::kF16    ? std::ldexp(1.0, -11)
                           : t == ConvType::kBF16 ? std::ldexp(1.0, -8)
                                                  : 0.0;
    const int64_t k_len = p.kh * p.kw * p.c;
    int failures = 0;
    for (int64_t n = 0; n < p.n; ++n) {
      for (int64_t oh = 0; oh < p.out_h; ++oh) {
        for (int64_t ow = 0; ow < p.out_w; ++ow) {
          for (int64_t o = 0; o < p.o && failures < 5; ++o) {
            double acc = 0, abs_acc = 0;
            for (int64_t kh = 0; kh < p.kh; ++kh) {
              const int64_t hd = oh * p.stride[0] - p.pad_lo[0] + kh * p.kdil[0];
              if (hd < 0 || hd % p.idil[0] != 0 || hd / p.idil[0] >= p.h) {
                continue;
              }
              for (int64_t kw = 0; kw < p.kw; ++kw) {
                const int64_t wd =
                    ow * p.stride[1] - p.pad_lo[1] + kw * p.kdil[1];
                if (wd < 0 || wd % p.idil[1] != 0 || wd / p.idil[1] >= p.w) {
                  continue;
                }
                const int64_t fh = p.flip ? p.kh - 1 - kh : kh;
                const int64_t fw = p.flip ? p.kw - 1 - kw : kw;
                for (int64_t c = 0; c < p.c; ++c) {
                  const double a = Load(
                      x, ((n * p.h + hd / p.idil[0]) * p.w + wd / p.idil[1]) *
                                 p.c + c,
                      t);
                  const double b =
                      Load(wt, ((o * p.kh + fh) * p.kw + fw) * p.c + c, t);
                  acc += a * b;
                  abs_acc += std::abs(a * b);
                }
              }
            }
            const int64_t i = ((n * p.out_h + oh) * p.out_w + ow) * p.o + o;
            const double got = Load(y, i, t);
            const double tol = 2.0 * k_len * u32 * abs_acc +
                               out_eps * std::abs(acc) + 1e-30;
            if (!(std::abs(got - acc) <= tol)) {
              ADD_FAILURE() << "out[" << n << "," << oh << "," << ow << ","
                            << o << "]: got " << got << " want " << acc
                            << " (tol " << tol << ")";
              ++failures;
            }
          }
        }
      }
    }
  }

  // Unfold of patch rows [row, row + rows) against the naive kernel, byte
  // for byte, with the input `in_shift` and the output `dst_shift` elements
  // into their buffers (a misaligned offset steps the vector width down).
  void CheckUnfold(const ConvParams& p, int64_t row, int64_t rows,
                   int64_t in_shift, int64_t dst_shift = 0) {
    SCOPED_TRACE(absl::StrCat("unfold ", ConvParamsDebugString(p), " rows ",
                              row, "+", rows, " shift ", in_shift, " dst ",
                              dst_shift));
    const int64_t item = Bytes(p.type);
    const int64_t n_in = p.n * p.h * p.w * p.c;
    const int64_t out_bytes = rows * p.kh * p.kw * p.c * item;
    std::vector<uint8_t> x = Random(p.type, n_in, 7 + n_in);
    absl::StatusOr<rt::Allocation> in =
        dev_->Allocate((n_in + in_shift) * item);
    absl::StatusOr<rt::Allocation> got =
        dev_->Allocate(out_bytes + dst_shift * item);
    absl::StatusOr<rt::Allocation> want = dev_->Allocate(out_bytes);
    ASSERT_THAT(in, IsOk());
    ASSERT_THAT(got, IsOk());
    ASSERT_THAT(want, IsOk());
    char* x_dev = static_cast<char*>(in->ptr) + in_shift * item;
    char* got_dev = static_cast<char*>(got->ptr) + dst_shift * item;
    std::memcpy(x_dev, x.data(), x.size());
    std::memset(got_dev, 0x7f, out_bytes);
    std::memset(want->ptr, 0x3c, out_bytes);
    ASSERT_THAT(Unfold(dev_.get(), stream_.get(), p, x_dev, got_dev, row,
                       rows),
                IsOk());
    absl::StatusOr<const rt::Kernel*> naive = dev_->GetKernel(
        kNaiveUnfoldMsl, absl::StrCat("naive_unfold_2d_", item));
    ASSERT_THAT(naive, IsOk());
    SteelConvParams sp = {};
    sp.N = p.n;
    sp.C = p.c;
    sp.O = p.o;
    sp.iS[0] = p.h;
    sp.iS[1] = p.w;
    sp.wS[0] = p.kh;
    sp.wS[1] = p.kw;
    sp.oS[0] = p.out_h;
    sp.oS[1] = p.out_w;
    for (int i = 0; i < 2; ++i) {
      sp.str[i] = p.stride[i];
      sp.pad[i] = p.pad_lo[i];
      sp.kdil[i] = p.kdil[i];
      sp.idil[i] = p.idil[i];
    }
    sp.in_strides[0] = p.h * p.w * p.c;
    sp.in_strides[1] = p.w * p.c;
    sp.in_strides[2] = p.c;
    sp.in_strides[3] = 1;
    sp.groups = 1;
    sp.flip = p.flip;
    const UnfoldRows ur{static_cast<int32_t>(row), static_cast<int32_t>(rows),
                        0};
    const int64_t cols = p.kh * p.kw * p.c;
    ASSERT_THAT(
        stream_->Launch(**naive,
                        {static_cast<uint32_t>((cols + 31) / 32),
                         static_cast<uint32_t>((rows + 7) / 8), 1},
                        {32, 8, 1},
                        {rt::KernelArg::Buffer(x_dev),
                         rt::KernelArg::Buffer(want->ptr),
                         rt::KernelArg::Bytes(&sp, sizeof(sp)),
                         rt::KernelArg::Bytes(&ur, sizeof(ur))}),
        IsOk());
    ASSERT_THAT(stream_->Synchronize(), IsOk());
    EXPECT_EQ(std::memcmp(got_dev, want->ptr, out_bytes), 0);
    for (void* ptr : {in->ptr, got->ptr, want->ptr}) {
      ASSERT_THAT(dev_->Deallocate(ptr), IsOk());
    }
  }

  // Runs the weight gradient `p` (kWeightGrad) with `plan` and compares
  // with the reference.
  void CheckWeightGrad(const ConvParams& p, const ConvPlan& plan) {
    SCOPED_TRACE(absl::StrCat("weight grad ", ConvParamsDebugString(p),
                              " rows ", plan.unfold_rows, " splits ",
                              plan.splits, " x ", plan.split_rows));
    const ConvType t = p.type;
    const int64_t n_in = p.n * p.h * p.w * p.c;
    const int64_t n_dy = p.n * p.out_h * p.out_w * p.o;
    const int64_t n_dw = p.o * p.kh * p.kw * p.c;
    std::vector<uint8_t> x = Random(t, n_in, 17 + n_in);
    std::vector<uint8_t> dy = Random(t, n_dy, 31 + n_dy);
    absl::StatusOr<rt::Allocation> in = dev_->Allocate(std::max<size_t>(x.size(), 1));
    absl::StatusOr<rt::Allocation> g = dev_->Allocate(std::max<size_t>(dy.size(), 1));
    absl::StatusOr<rt::Allocation> out = dev_->Allocate(n_dw * Bytes(t));
    ASSERT_THAT(in, IsOk());
    ASSERT_THAT(g, IsOk());
    ASSERT_THAT(out, IsOk());
    void* ws = nullptr;
    if (plan.workspace_bytes > 0) {
      absl::StatusOr<rt::Allocation> a = dev_->Allocate(plan.workspace_bytes);
      ASSERT_THAT(a, IsOk());
      ws = a->ptr;
      std::memset(ws, 0x7f, plan.workspace_bytes);  // stale values show
    }
    std::memcpy(in->ptr, x.data(), x.size());
    std::memcpy(g->ptr, dy.data(), dy.size());
    std::memset(out->ptr, 0xff, n_dw * Bytes(t));  // missing writes show
    ASSERT_THAT(RunConv(dev_.get(), stream_.get(), p, plan, in->ptr, g->ptr,
                        out->ptr, ws),
                IsOk());
    ASSERT_THAT(stream_->Synchronize(), IsOk());
    std::vector<uint8_t> y(n_dw * Bytes(t));
    std::memcpy(y.data(), out->ptr, y.size());
    for (void* ptr : {in->ptr, g->ptr, out->ptr, ws}) {
      if (ptr != nullptr) ASSERT_THAT(dev_->Deallocate(ptr), IsOk());
    }

    const double u32 = std::ldexp(1.0, -24);
    const double out_eps = t == ConvType::kF16    ? std::ldexp(1.0, -11)
                           : t == ConvType::kBF16 ? std::ldexp(1.0, -8)
                                                  : 0.0;
    const int64_t m = p.n * p.out_h * p.out_w;
    int failures = 0;
    for (int64_t o = 0; o < p.o; ++o) {
      for (int64_t fh = 0; fh < p.kh; ++fh) {
        for (int64_t fw = 0; fw < p.kw; ++fw) {
          // The tap that weight element (fh, fw) multiplies.
          const int64_t kh = p.flip ? p.kh - 1 - fh : fh;
          const int64_t kw = p.flip ? p.kw - 1 - fw : fw;
          for (int64_t c = 0; c < p.c && failures < 5; ++c) {
            double acc = 0, abs_acc = 0;
            for (int64_t n = 0; n < p.n; ++n) {
              for (int64_t oh = 0; oh < p.out_h; ++oh) {
                const int64_t hd =
                    oh * p.stride[0] - p.pad_lo[0] + kh * p.kdil[0];
                if (hd < 0 || hd % p.idil[0] != 0 || hd / p.idil[0] >= p.h) {
                  continue;
                }
                for (int64_t ow = 0; ow < p.out_w; ++ow) {
                  const int64_t wd =
                      ow * p.stride[1] - p.pad_lo[1] + kw * p.kdil[1];
                  if (wd < 0 || wd % p.idil[1] != 0 ||
                      wd / p.idil[1] >= p.w) {
                    continue;
                  }
                  const double a = Load(
                      x, ((n * p.h + hd / p.idil[0]) * p.w + wd / p.idil[1]) *
                                 p.c + c,
                      t);
                  const double b =
                      Load(dy, ((n * p.out_h + oh) * p.out_w + ow) * p.o + o,
                           t);
                  acc += a * b;
                  abs_acc += std::abs(a * b);
                }
              }
            }
            const int64_t i = ((o * p.kh + fh) * p.kw + fw) * p.c + c;
            const double got = Load(y, i, t);
            const double tol = 2.0 * (m + plan.splits) * u32 * abs_acc +
                               out_eps * std::abs(acc) + 1e-30;
            if (!(std::abs(got - acc) <= tol)) {
              ADD_FAILURE() << "dw[" << o << "," << fh << "," << fw << ","
                            << c << "]: got " << got << " want " << acc
                            << " (tol " << tol << ")";
              ++failures;
            }
          }
        }
      }
    }
  }

  // The weight gradient of `k`'s convolution in every type, as planned and
  // with launches bounded to `max_launch_flops` (several chunks).
  void CheckWeightGrads(const Case& k, uint64_t max_launch_flops = 0) {
    for (ConvType t : kTypes) {
      ConvParams p = Params(k, t);
      p.kind = ConvKind::kWeightGrad;
      absl::StatusOr<ConvPlan> plan = PlanConv(p);
      ASSERT_THAT(plan, IsOk()) << ConvParamsDebugString(p);
      EXPECT_EQ(plan->path, ConvPath::kWeightGrad);
      CheckWeightGrad(p, *plan);
      if (max_launch_flops > 0) {
        absl::StatusOr<ConvPlan> small =
            PlanConv(p, nullptr, max_launch_flops);
        ASSERT_THAT(small, IsOk());
        EXPECT_LT(small->unfold_rows, p.n * p.out_h * p.out_w);
        CheckWeightGrad(p, *small);
      }
    }
  }

  // Every path PlanConv accepts for `k`, in every type.
  void CheckAllPaths(const Case& k) {
    for (ConvType t : kTypes) {
      const ConvParams p = Params(k, t);
      absl::StatusOr<ConvPlan> chosen = PlanConv(p);
      ASSERT_THAT(chosen, IsOk()) << ConvParamsDebugString(p);
      Check(p, *chosen);
      for (ConvPath path : kPaths) {
        absl::StatusOr<ConvPlan> forced = PlanConv(p, &path);
        if (!forced.ok()) {
          EXPECT_THAT(forced, StatusIs(absl::StatusCode::kUnimplemented));
          continue;
        }
        if (path != chosen->path) Check(p, *forced);
      }
    }
  }

  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
};

// The cnn bench's layers (bench/jax_bench.py, batch 4 here) and the input
// gradient of its second layer as JAX emits it: the cotangent dilated by
// the stride, padded (2, 1), correlated with the flipped kernel.
TEST_F(ConvTest, CnnBench) {
  Case conv1{4, 32, 32, 3, 32, 3, 3};
  conv1.pad_lo[0] = conv1.pad_lo[1] = conv1.pad_hi[0] = conv1.pad_hi[1] = 1;
  CheckAllPaths(conv1);
  Case conv2{4, 32, 32, 32, 64, 3, 3};
  conv2.stride[0] = conv2.stride[1] = 2;
  conv2.pad_hi[0] = conv2.pad_hi[1] = 1;
  CheckAllPaths(conv2);
  // Input gradient of conv2: [4, 16, 16, 64] -> [4, 32, 32, 32]; weight as
  // [O = 32, kH, kW, C = 64].
  Case igrad{4, 16, 16, 64, 32, 3, 3};
  igrad.idil[0] = igrad.idil[1] = 2;
  igrad.pad_lo[0] = igrad.pad_lo[1] = 2;
  igrad.pad_hi[0] = igrad.pad_hi[1] = 1;
  igrad.flip = true;
  CheckAllPaths(igrad);
}

// Channel counts around the specializations: C <= 4 (small-channel
// loaders), C % 16, unaligned C (pad-channels / general / explicit); O <= 16
// (8-wide tile), O % 16, O 17.
TEST_F(ConvTest, Channels) {
  for (int64_t c : {1, 2, 3, 4, 5, 16, 32}) {
    for (int64_t o : {8, 16, 17, 64}) {
      Case k{2, 9, 11, c, o, 3, 3};
      k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
      CheckAllPaths(k);
    }
  }
}

// Output channels the implicit kernel pads (20, 24, 31 -> 32; 33 -> 64),
// stride 1 and 2, flipped (input gradients), small and 64-wide tiles, the
// small-channel (C = 3) loader.
TEST_F(ConvTest, PaddedOutputChannels) {
  for (int64_t o : {20, 24, 31, 33}) {
    for (int64_t c : {3, 16, 64}) {
      Case k{2, 15, 13, c, o, 3, 3};
      k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
      k.flip = o % 2 == 0;
      if (o == 31) k.stride[0] = k.stride[1] = 2;
      const ConvParams p = Params(k, ConvType::kF32);
      absl::StatusOr<ConvPlan> plan = PlanConv(p);
      ASSERT_THAT(plan, IsOk());
      EXPECT_EQ(plan->path, ConvPath::kImplicit) << o;
      EXPECT_EQ(plan->padded_o, o <= 32 ? 32 : 64) << o;
      CheckAllPaths(k);
    }
  }
  // 8192+ output pixels and C = 64: the 64-row tile, 32 columns.
  Case big{40, 16, 16, 64, 24, 3, 3};
  big.pad_lo[0] = big.pad_lo[1] = big.pad_hi[0] = big.pad_hi[1] = 1;
  big.flip = true;
  absl::StatusOr<ConvPlan> plan = PlanConv(Params(big, ConvType::kBF16));
  ASSERT_THAT(plan, IsOk());
  EXPECT_EQ(plan->tile, (ConvTile{64, 32, 16, 2, 2}));
  CheckAllPaths(big);
  // O * K within 32 bits but the padded weight's 32 * K not (K = 16 *
  // 4194305, 256 output pixels): the general kernel, not padded.
  Case wide{1, 1, 1, 16, 24, 1, 4194305};
  wide.pad_lo[1] = 2097280;
  wide.pad_hi[1] = 2097279;
  plan = PlanConv(Params(wide, ConvType::kF32));
  ASSERT_THAT(plan, IsOk());
  EXPECT_EQ(plan->path, ConvPath::kGeneral);
  EXPECT_EQ(plan->padded_o, 0);
}

// Strides, kernel dilation, asymmetric and zero padding, input dilation
// with and without flip, a 1-D convolution as H = 1.
TEST_F(ConvTest, Geometry) {
  Case strided{3, 13, 10, 16, 32, 3, 2};
  strided.stride[0] = 2;
  strided.stride[1] = 3;
  strided.pad_lo[0] = 1;
  strided.pad_hi[1] = 2;
  CheckAllPaths(strided);
  Case dilated{2, 12, 12, 32, 16, 3, 3};
  dilated.kdil[0] = 2;
  dilated.kdil[1] = 3;
  dilated.pad_lo[0] = 2;
  dilated.pad_hi[0] = 1;
  dilated.pad_lo[1] = 3;
  CheckAllPaths(dilated);
  for (bool flip : {false, true}) {
    Case idil{2, 7, 6, 16, 16, 3, 3};
    idil.idil[0] = 2;
    idil.idil[1] = 3;
    idil.pad_lo[0] = 2;
    idil.pad_hi[0] = 1;
    idil.pad_lo[1] = 1;
    idil.pad_hi[1] = 2;
    idil.stride[1] = 2;
    idil.flip = flip;
    CheckAllPaths(idil);
  }
  Case one_d{3, 1, 50, 16, 32, 1, 5};
  one_d.pad_lo[1] = 2;
  one_d.pad_hi[1] = 2;
  CheckAllPaths(one_d);
  Case flip_small{2, 10, 10, 32, 32, 3, 3};
  flip_small.pad_lo[0] = flip_small.pad_lo[1] = 2;
  flip_small.flip = true;
  CheckAllPaths(flip_small);
}

// A kernel wider or taller than the small-filter bit masks (kW or kH 17 >
// 16): the specialized kernel's large-filter input loader, with and without
// flip (whose bounds check MLX gets wrong) and an asymmetric kernel.
TEST_F(ConvTest, LargeFilter) {
  for (bool flip : {false, true}) {
    Case k{2, 6, 24, 16, 16, 3, 17};
    k.pad_lo[0] = 1;
    k.pad_hi[0] = 2;
    k.pad_lo[1] = 3;
    k.pad_hi[1] = 10;
    k.flip = flip;
    CheckAllPaths(k);
    Case tall{2, 24, 6, 16, 16, 17, 3};
    tall.pad_lo[0] = 3;
    tall.pad_hi[0] = 10;
    tall.pad_lo[1] = 1;
    tall.pad_hi[1] = 2;
    tall.flip = flip;
    CheckAllPaths(tall);
  }
}

// Enough output pixels (M = 8192) and channels (C = 64) for the 64-row
// tiles (ImplicitTile, GeneralTile).
TEST_F(ConvTest, LargeTiles) {
  Case k{2, 64, 64, 64, 64, 3, 3};
  k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
  const ConvParams p = Params(k, ConvType::kF32);
  absl::StatusOr<ConvPlan> plan = PlanConv(p);
  ASSERT_THAT(plan, IsOk());
  EXPECT_EQ(plan->tile.bm, 64);
  const ConvPath general = ConvPath::kGeneral;
  absl::StatusOr<ConvPlan> gplan = PlanConv(p, &general);
  ASSERT_THAT(gplan, IsOk());
  EXPECT_EQ(gplan->tile.bm, 64);
  CheckAllPaths(k);
}

// Every path with its launches bounded to a fifth of the convolution's
// flops: several launches of row tiles (implicit, pad-channels, general) or
// unfold chunks (explicit).
TEST_F(ConvTest, LaunchSplits) {
  Case k{3, 12, 10, 16, 32, 3, 3};
  k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
  Case idil = k;
  idil.idil[0] = 2;
  idil.stride[1] = 2;
  idil.flip = true;
  for (const Case& c : {k, idil}) {
    for (ConvType t : kTypes) {
      const ConvParams p = Params(c, t);
      const uint64_t limit = ConvFlops(p) / 5;
      for (ConvPath path : kPaths) {
        absl::StatusOr<ConvPlan> plan = PlanConv(p, &path, limit);
        if (!plan.ok()) continue;
        if (path == ConvPath::kExplicit) {
          EXPECT_LT(plan->unfold_rows, p.n * p.out_h * p.out_w);
        } else {
          EXPECT_GT(plan->launch_tiles_m, 0);
          EXPECT_LT(plan->launch_tiles_m, plan->tiles_m);
        }
        Check(p, *plan);
      }
    }
  }
}

// The weight gradient on the cnn bench's layers (batch 4) and a spread of
// geometries: strides, kernel and input dilation, asymmetric padding, flip,
// small and unaligned channels, a 1-D convolution; one part and several
// (split-K), one chunk and several.
TEST_F(ConvTest, WeightGrad) {
  Case conv1{4, 32, 32, 3, 32, 3, 3};
  conv1.pad_lo[0] = conv1.pad_lo[1] = conv1.pad_hi[0] = conv1.pad_hi[1] = 1;
  CheckWeightGrads(conv1, 2ull * 32 * 27 * 1000);
  Case conv2{4, 32, 32, 32, 64, 3, 3};
  conv2.stride[0] = conv2.stride[1] = 2;
  conv2.pad_hi[0] = conv2.pad_hi[1] = 1;
  CheckWeightGrads(conv2, 2ull * 64 * 288 * 300);
  Case geo{3, 13, 10, 5, 17, 3, 2};
  geo.stride[0] = 2;
  geo.kdil[1] = 2;
  geo.pad_lo[0] = 1;
  geo.pad_hi[1] = 2;
  CheckWeightGrads(geo, 2ull * 17 * 30 * 7);
  Case idil{2, 7, 6, 16, 16, 3, 3};
  idil.idil[0] = 2;
  idil.idil[1] = 3;
  idil.pad_lo[0] = 2;
  idil.pad_hi[0] = 1;
  idil.pad_lo[1] = 1;
  idil.flip = true;
  CheckWeightGrads(idil);
  Case one_d{3, 1, 50, 16, 32, 1, 5};
  one_d.pad_lo[1] = 2;
  one_d.pad_hi[1] = 2;
  CheckWeightGrads(one_d);
  // One part: many (O x K) tiles already.
  Case wide{2, 8, 8, 64, 128, 3, 3};
  wide.pad_lo[0] = wide.pad_lo[1] = wide.pad_hi[0] = wide.pad_hi[1] = 1;
  absl::StatusOr<ConvPlan> plan = [&] {
    ConvParams p = Params(wide, ConvType::kF32);
    p.kind = ConvKind::kWeightGrad;
    return PlanConv(p);
  }();
  ASSERT_THAT(plan, IsOk());
  EXPECT_EQ(plan->splits, 1);
  CheckWeightGrads(wide);
}

// The weight gradient's plan: split-K parts on the cnn bench's layers,
// bounded workspace, dW zero-filled for an empty batch.
TEST_F(ConvTest, WeightGradPlan) {
  Case conv1{32, 32, 32, 3, 32, 3, 3};
  conv1.pad_lo[0] = conv1.pad_lo[1] = conv1.pad_hi[0] = conv1.pad_hi[1] = 1;
  ConvParams p = Params(conv1, ConvType::kF32);
  p.kind = ConvKind::kWeightGrad;
  absl::StatusOr<ConvPlan> plan = PlanConv(p);
  ASSERT_THAT(plan, IsOk());
  EXPECT_EQ(plan->unfold_rows, 32 * 32 * 32);
  EXPECT_GT(plan->splits, 1);
  EXPECT_GE(plan->split_rows, 256);
  EXPECT_EQ(plan->tile, (ConvTile{32, 32, 16, 2, 2}));  // O = 32, K = 27
  // The airbench94 CIFAR-10 layers (bf16, batch 1024, 3x3 SAME): 64 x 64
  // tiles, >= 512 threadgroups unless the partials cap (8 MiB) binds, and
  // the patches plus partials within the 64 MiB chunk bound.
  struct Layer {
    int64_t hw, c, o, splits;
  };
  for (const Layer& l : {Layer{31, 24, 64, 128}, Layer{15, 64, 64, 56},
                         Layer{15, 64, 256, 14}, Layer{7, 256, 256, 3},
                         Layer{3, 256, 256, 3}}) {
    Case k{1024, l.hw, l.hw, l.c, l.o, 3, 3};
    k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
    ConvParams lp = Params(k, ConvType::kBF16);
    lp.kind = ConvKind::kWeightGrad;
    SCOPED_TRACE(ConvParamsDebugString(lp));
    absl::StatusOr<ConvPlan> lplan = PlanConv(lp);
    ASSERT_THAT(lplan, IsOk());
    EXPECT_EQ(lplan->tile, (ConvTile{64, 64, 16, 2, 2}));
    EXPECT_EQ(lplan->splits, l.splits);
    EXPECT_LE(lplan->splits * l.o * 9 * l.c * 4, 8 << 20);
    EXPECT_LE(lplan->workspace_bytes, 64u << 20);
  }
  const ConvPath implicit = ConvPath::kImplicit;
  EXPECT_THAT(PlanConv(p, &implicit),
              StatusIs(absl::StatusCode::kUnimplemented));
  ConvParams fwd = Params(conv1, ConvType::kF32);
  const ConvPath wgrad = ConvPath::kWeightGrad;
  EXPECT_THAT(PlanConv(fwd, &wgrad),
              StatusIs(absl::StatusCode::kUnimplemented));

  ConvParams empty = Params(Case{0, 8, 8, 16, 16, 3, 3}, ConvType::kF32);
  empty.kind = ConvKind::kWeightGrad;
  absl::StatusOr<ConvPlan> eplan = PlanConv(empty);
  ASSERT_THAT(eplan, IsOk());
  absl::StatusOr<rt::Allocation> dw = dev_->Allocate(16 * 9 * 16 * 4);
  ASSERT_THAT(dw, IsOk());
  std::memset(dw->ptr, 0xff, 16 * 9 * 16 * 4);
  ASSERT_THAT(RunConv(dev_.get(), stream_.get(), empty, *eplan, nullptr,
                      nullptr, dw->ptr, nullptr),
              IsOk());
  ASSERT_THAT(stream_->Synchronize(), IsOk());
  const std::vector<uint8_t> zeros(16 * 9 * 16 * 4, 0);
  EXPECT_EQ(std::memcmp(dw->ptr, zeros.data(), zeros.size()), 0);
  ASSERT_THAT(dev_->Deallocate(dw->ptr), IsOk());
}

// f16/bf16 weight gradients whose patch rows * kH * kW * C is odd (an odd
// number of 2-byte elements before the f32 partials): the partials must
// still be 4-byte aligned for the GEMM's stores, sum_splits' loads and the
// buffer offset. 61 x 61 = 3721 output pixels, k = 27; in one chunk and in
// chunks of 1001 rows.
TEST_F(ConvTest, WeightGradOddPatchBytes) {
  const Case k{1, 63, 63, 3, 32, 3, 3};
  for (ConvType t : kTypes) {
    ConvParams p = Params(k, t);
    p.kind = ConvKind::kWeightGrad;
    for (uint64_t max_flops : {kMaxLaunchFlops, 2ull * 32 * 27 * 1001}) {
      absl::StatusOr<ConvPlan> plan = PlanConv(p, nullptr, max_flops);
      ASSERT_THAT(plan, IsOk());
      EXPECT_EQ(plan->unfold_rows % 2, 1);
      EXPECT_GT(plan->splits, 1);  // not direct: the partials are used
      EXPECT_GE(plan->workspace_bytes,
                (plan->unfold_rows * 27 * Bytes(t) + 255) / 256 * 256 +
                    plan->splits * 32 * 27 * 4);
      CheckWeightGrad(p, *plan);
    }
  }
}

// PlanConv refuses what the explicit and weight-gradient GEMMs cannot run,
// so the rewriter leaves those convolutions to the loop emitter instead of
// failing at run time: SteelGemmSupports' leading-dimension limit (kH * kW *
// C, and O), checked here to agree with the steel GEMM's own.
TEST_F(ConvTest, SteelLeadingDimensionLimit) {
  constexpr int64_t kMaxLd = std::numeric_limits<int32_t>::max() / 256;
  blas::GemmParams g;
  g.m = g.n = g.k = 1;
  g.a.ld = g.b.ld = g.c.ld = kMaxLd;
  EXPECT_TRUE(blas::SteelGemmSupports(g));
  g.b.ld = kMaxLd + 1;
  EXPECT_FALSE(blas::SteelGemmSupports(g));

  const ConvPath explicit_path = ConvPath::kExplicit;
  for (int64_t n : {kMaxLd, kMaxLd + 1}) {
    const bool ok = n <= kMaxLd;
    // A 1-D filter over n taps (one output pixel), and n output channels.
    for (const Case& k : {Case{1, 1, n, 1, 1, 1, n}, Case{1, 1, 1, 1, n, 1, 1}}) {
      ConvParams fwd = Params(k, ConvType::kF32);
      ConvParams wgrad = fwd;
      wgrad.kind = ConvKind::kWeightGrad;
      SCOPED_TRACE(ConvParamsDebugString(fwd));
      for (absl::StatusOr<ConvPlan> plan :
           {PlanConv(wgrad), PlanConv(fwd, &explicit_path)}) {
        if (ok) {
          EXPECT_THAT(plan, IsOk());
        } else {
          EXPECT_THAT(plan, StatusIs(absl::StatusCode::kUnimplemented));
        }
      }
    }
  }
}

// The explicit path over several unfold tiles (the last one partial).
TEST_F(ConvTest, ExplicitTiles) {
  Case k{3, 9, 7, 5, 17, 3, 3};
  k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
  for (ConvType t : kTypes) {
    const ConvParams p = Params(k, t);
    const ConvPath path = ConvPath::kExplicit;
    absl::StatusOr<ConvPlan> plan = PlanConv(p, &path);
    ASSERT_THAT(plan, IsOk());
    plan->unfold_rows = 29;  // 189 output pixels -> 7 tiles
    plan->workspace_bytes = 29 * p.kh * p.kw * p.c * Bytes(t);
    Check(p, *plan);
  }
}

// The vectorized unfold is the naive one, byte for byte: channel counts
// for every vector width (16 bytes down to one element: C 1, 3, 5, 6, 12,
// 24 in 2- and 4-byte types), input and output offsets that step the width
// down, odd widths, strides, kernel and input dilation, flip, asymmetric
// and negative (cropping) padding, a single tap; the whole patch matrix and
// a chunk from a row offset.
TEST_F(ConvTest, Unfold) {
  std::vector<Case> cases;
  for (int64_t c : {1, 3, 5, 6, 12, 24}) {
    Case k{2, 7, 9, c, 4, 3, 3};
    k.pad_lo[0] = k.pad_lo[1] = k.pad_hi[0] = k.pad_hi[1] = 1;
    cases.push_back(k);
  }
  Case geo{3, 13, 11, 24, 4, 3, 2};
  geo.stride[0] = 2;
  geo.stride[1] = 3;
  geo.kdil[1] = 2;
  geo.pad_lo[0] = 1;
  geo.pad_hi[0] = 2;
  geo.pad_hi[1] = 3;
  cases.push_back(geo);
  for (bool flip : {false, true}) {
    Case idil{2, 7, 5, 12, 4, 3, 3};
    idil.idil[0] = 2;
    idil.idil[1] = 3;
    idil.pad_lo[0] = 2;
    idil.pad_hi[0] = 1;
    idil.pad_lo[1] = 1;
    idil.pad_hi[1] = 2;
    idil.stride[1] = 2;
    idil.flip = flip;
    cases.push_back(idil);
  }
  Case crop{2, 9, 8, 12, 4, 3, 3};  // negative high padding
  crop.pad_lo[1] = 1;
  crop.pad_hi[0] = -2;
  crop.pad_hi[1] = -1;
  cases.push_back(crop);
  Case one_tap{2, 9, 7, 24, 4, 1, 1};
  one_tap.stride[0] = 2;
  cases.push_back(one_tap);
  for (const Case& k : cases) {
    for (ConvType t : kTypes) {
      const ConvParams p = Params(k, t);
      const int64_t m = p.n * p.out_h * p.out_w;
      for (int64_t shift : {0, 1, 2, 4}) {
        CheckUnfold(p, 0, m, shift);
        CheckUnfold(p, 0, m, 0, shift);
      }
      CheckUnfold(p, m / 3, m - m / 3 - 1, 0);
    }
  }
  // One launch past 32-bit indexing is refused (the kernel's index would
  // wrap), before any buffer is touched.
  const ConvParams big = Params(cases[0], ConvType::kF32);
  const int64_t too_many = (int64_t{1} << 31) / (9 * big.c) + 1;
  EXPECT_THAT(Unfold(dev_.get(), stream_.get(), big, nullptr, nullptr, 0,
                     too_many),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// What PlanConv picks (MLX's decision tree without Winograd), and refuses.
TEST_F(ConvTest, Plan) {
  auto path_of = [](const Case& k) {
    absl::StatusOr<ConvPlan> plan = PlanConv(Params(k, ConvType::kF32));
    EXPECT_THAT(plan, IsOk());
    return plan.ok() ? plan->path : ConvPath::kExplicit;
  };
  Case conv1{32, 32, 32, 3, 32, 3, 3};
  conv1.pad_lo[0] = conv1.pad_lo[1] = conv1.pad_hi[0] = conv1.pad_hi[1] = 1;
  EXPECT_EQ(path_of(conv1), ConvPath::kImplicit);
  absl::StatusOr<ConvPlan> plan1 = PlanConv(Params(conv1, ConvType::kF32));
  ASSERT_THAT(plan1, IsOk());
  EXPECT_EQ(plan1->n_channels, 3);
  EXPECT_EQ(plan1->tile, (ConvTile{32, 32, 16, 2, 2}));
  Case conv2{32, 32, 32, 32, 64, 3, 3};
  conv2.stride[0] = conv2.stride[1] = 2;
  conv2.pad_hi[0] = conv2.pad_hi[1] = 1;
  absl::StatusOr<ConvPlan> plan2 = PlanConv(Params(conv2, ConvType::kF32));
  ASSERT_THAT(plan2, IsOk());
  EXPECT_EQ(plan2->path, ConvPath::kImplicit);
  EXPECT_TRUE(plan2->small_filter);
  EXPECT_EQ(plan2->tile, (ConvTile{32, 64, 16, 2, 2}));
  Case igrad{32, 16, 16, 64, 32, 3, 3};
  igrad.idil[0] = igrad.idil[1] = 2;
  igrad.pad_lo[0] = igrad.pad_lo[1] = 2;
  igrad.pad_hi[0] = igrad.pad_hi[1] = 1;
  igrad.flip = true;
  EXPECT_EQ(path_of(igrad), ConvPath::kGeneral);
  // Unaligned C, stride 1, 3x3, large output: pad the channels.
  Case pad{2, 16, 16, 5, 16, 3, 3};
  pad.pad_lo[0] = pad.pad_lo[1] = pad.pad_hi[0] = pad.pad_hi[1] = 1;
  EXPECT_EQ(path_of(pad), ConvPath::kPadChannels);
  // Its kernels run over the padded channels (5 -> 16, 3.2x the flops), and
  // its launches are bounded by those.
  const ConvParams pp = Params(pad, ConvType::kF32);
  absl::StatusOr<ConvPlan> pad_plan = PlanConv(pp, nullptr, ConvFlops(pp));
  ASSERT_THAT(pad_plan, IsOk());
  EXPECT_EQ(pad_plan->path, ConvPath::kPadChannels);
  EXPECT_LE(pad_plan->launch_tiles_m * 3, pad_plan->tiles_m);
  // O not a whole number of column tiles: the implicit kernel on the
  // weight with its O rows zero-padded in the workspace (MLX's specialized
  // kernel reads O % 16 weights past the buffer, the rest go general).
  Case o48{2, 16, 16, 16, 48, 3, 3};
  EXPECT_EQ(path_of(o48), ConvPath::kImplicit);
  absl::StatusOr<ConvPlan> o48_plan = PlanConv(Params(o48, ConvType::kF32));
  ASSERT_THAT(o48_plan, IsOk());
  EXPECT_EQ(o48_plan->padded_o, 64);
  EXPECT_EQ(o48_plan->workspace_bytes, 64u * 3 * 3 * 16 * 4);
  // airbench94's 31x31 64 -> 24 input gradient (bf16, N = 1024): 32-wide
  // columns (24 of 32 used, not 24 of 64), flops charged for 32.
  Case g1c1{1024, 31, 31, 64, 24, 3, 3};
  g1c1.pad_lo[0] = g1c1.pad_lo[1] = g1c1.pad_hi[0] = g1c1.pad_hi[1] = 1;
  g1c1.flip = true;
  const ConvParams gp = Params(g1c1, ConvType::kBF16);
  absl::StatusOr<ConvPlan> g1c1_plan = PlanConv(gp);
  ASSERT_THAT(g1c1_plan, IsOk());
  EXPECT_EQ(g1c1_plan->path, ConvPath::kImplicit);
  EXPECT_EQ(g1c1_plan->tile, (ConvTile{64, 32, 16, 2, 2}));
  EXPECT_EQ(g1c1_plan->padded_o, 32);
  // Launches bounded by the true flops would cover it in one; charged for
  // 32 columns they take two.
  absl::StatusOr<ConvPlan> bounded = PlanConv(gp, nullptr, ConvFlops(gp));
  ASSERT_THAT(bounded, IsOk());
  EXPECT_LT(bounded->launch_tiles_m, bounded->tiles_m);
  // Unaligned channels, small output: explicit.
  Case small{1, 4, 4, 5, 17, 3, 3};
  EXPECT_EQ(path_of(small), ConvPath::kExplicit);

  ConvParams grouped = Params(conv2, ConvType::kF32);
  grouped.groups = 2;
  EXPECT_THAT(PlanConv(grouped), StatusIs(absl::StatusCode::kUnimplemented));
  ConvParams neg = Params(conv2, ConvType::kF32);
  neg.pad_lo[0] = -1;
  EXPECT_THAT(PlanConv(neg), StatusIs(absl::StatusCode::kUnimplemented));
  ConvParams empty = Params(conv2, ConvType::kF32);
  empty.c = 0;
  EXPECT_THAT(PlanConv(empty), StatusIs(absl::StatusCode::kUnimplemented));
  ConvParams bad = Params(conv2, ConvType::kF32);
  bad.stride[0] = 0;
  EXPECT_THAT(PlanConv(bad), StatusIs(absl::StatusCode::kInvalidArgument));
  ConvParams huge = Params(conv2, ConvType::kF32);
  huge.n = 1 << 20;
  EXPECT_THAT(PlanConv(huge), StatusIs(absl::StatusCode::kUnimplemented));
}

// An empty output launches nothing (no buffers needed).
TEST_F(ConvTest, EmptyOutput) {
  ConvParams p = Params(Case{0, 8, 8, 16, 16, 3, 3}, ConvType::kF32);
  absl::StatusOr<ConvPlan> plan = PlanConv(p);
  ASSERT_THAT(plan, IsOk());
  EXPECT_THAT(RunConv(dev_.get(), stream_.get(), p, *plan, nullptr, nullptr,
                      nullptr, nullptr),
              IsOk());
}

}  // namespace
}  // namespace conv
}  // namespace metal_pjrt
