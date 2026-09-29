// RunConv (conv.h) against a double reference on rt::Device alone (no XLA):
// for every case, every path PlanConv accepts (the one it picks, and each
// forced one), in f32, f16 and bf16. Cases: the cnn bench's layers and its
// input gradient (input dilation + flip), channel counts around the kernels'
// specializations (C 1-5, 16, 32; O 8, 16, 17, 64), strides, kernel
// dilation, asymmetric padding, a kernel wider than the small-filter masks
// (with flip), several unfold tiles on the explicit path, and the path
// choice itself. f16/bf16 accumulate in f32, so all three types are within
// the f32 accumulation error of the reference on the rounded inputs, plus
// the output rounding. Needs a Metal device.
#include "metal_pjrt/conv/conv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

// A kernel wider than the small-filter bit masks (kW 17 > 16): the
// specialized kernel's large-filter input loader, with and without flip
// (whose bounds check MLX gets wrong) and an asymmetric kernel.
TEST_F(ConvTest, LargeFilter) {
  for (bool flip : {false, true}) {
    Case k{2, 6, 24, 16, 16, 3, 17};
    k.pad_lo[0] = 1;
    k.pad_hi[0] = 2;
    k.pad_lo[1] = 3;
    k.pad_hi[1] = 10;
    k.flip = flip;
    CheckAllPaths(k);
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
  // O % 16 == 0 but not a whole number of 32-wide tiles: general (MLX's
  // specialized kernel would read weight rows past the buffer).
  Case o48{2, 16, 16, 16, 48, 3, 3};
  EXPECT_EQ(path_of(o48), ConvPath::kGeneral);
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
