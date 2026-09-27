// Tests RunSteelGemm through a metal_pjrt::rt::Stream against a CPU double
// reference (no XLA). Needs a Metal device.
#include "metal_pjrt_plugin/blas/steel_gemm.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/blas/mps_gemm.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace blas {
namespace {

uint16_t F32ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}
float Bf16ToF32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t F32ToF16(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
float F16ToF32(uint16_t u) {
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return static_cast<float>(h);
}
void Store(void* base, int64_t i, MpsDType t, float v) {
  switch (t) {
    case MpsDType::kF32: static_cast<float*>(base)[i] = v; break;
    case MpsDType::kF16: static_cast<uint16_t*>(base)[i] = F32ToF16(v); break;
    case MpsDType::kBF16: static_cast<uint16_t*>(base)[i] = F32ToBf16(v); break;
  }
}
float Load(const void* base, int64_t i, MpsDType t) {
  switch (t) {
    case MpsDType::kF32: return static_cast<const float*>(base)[i];
    case MpsDType::kF16: return F16ToF32(static_cast<const uint16_t*>(base)[i]);
    case MpsDType::kBF16: return Bf16ToF32(static_cast<const uint16_t*>(base)[i]);
  }
  return 0;
}
double Eps(MpsDType t) {
  switch (t) {
    case MpsDType::kF32: return 1e-5;
    case MpsDType::kF16: return 1.0 / 1024;
    case MpsDType::kBF16: return 1.0 / 128;
  }
  return 0;
}

struct Case {
  int64_t m, n, k, batch = 1;
  MpsDType in = MpsDType::kBF16, out = MpsDType::kBF16;
  bool ta = false, tb = false;
  double alpha = 1.0, beta = 0.0;
  int64_t pad = 0;           // extra elements per stored row
  bool broadcast_b = false;  // B batch stride 0
  const SteelTile* tile = nullptr;
  std::string Name() const {
    return absl::StrCat(MpsDTypeName(in), "->", MpsDTypeName(out), " ", m,
                        "x", n, "x", k, " b", batch, ta ? " T" : " N",
                        tb ? "T" : "N", " a", alpha, " b", beta, " pad", pad,
                        broadcast_b ? " bcastB" : "",
                        tile ? absl::StrCat(" tile", tile->bm, "x", tile->bn,
                                            "x", tile->bk, "w", tile->wm, "x",
                                            tile->wn)
                             : "");
  }
};

class SteelGemmTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    auto d = rt::Device::Create(0);
    ASSERT_TRUE(d.ok()) << d.status();
    device_ = d->release();
    auto s = device_->CreateStream();
    ASSERT_TRUE(s.ok()) << s.status();
    stream_ = s->release();
  }

  struct Op {
    rt::Allocation alloc;
    int64_t rows, cols, ld, stride;
  };

  Op MakeOp(int64_t rows, int64_t cols, int64_t pad, int64_t batches,
            bool broadcast, MpsDType t, std::mt19937* rng) {
    Op op{{}, rows, cols, cols + pad, broadcast ? 0 : rows * (cols + pad)};
    const int64_t nb = broadcast ? 1 : batches;
    // Exactly sized: the last row has no padding (like XLA buffers).
    const int64_t elems = std::max<int64_t>(
        1, (nb - 1) * op.stride + (rows > 0 ? (rows - 1) * op.ld + cols : 0));
    auto a = device_->Allocate(elems * MpsDTypeSize(t));
    EXPECT_TRUE(a.ok()) << a.status();
    op.alloc = *a;
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int64_t i = 0; i < elems; ++i) Store(op.alloc.ptr, i, t, u(*rng));
    return op;
  }

  MpsOperand Bind(const Op& op, MpsDType t, bool trans) {
    MpsOperand o;
    auto ref = device_->Resolve(op.alloc.ptr);
    EXPECT_TRUE(ref.ok());
    o.buffer = ref->buffer;
    o.offset = ref->offset;
    o.ld = op.ld;
    o.batch_stride = op.stride;
    o.dtype = t;
    o.transpose = trans;
    return o;
  }

  void Run(const Case& c) {
    SCOPED_TRACE(c.Name());
    std::mt19937 rng(1234);
    Op a = c.ta ? MakeOp(c.k, c.m, c.pad, c.batch, false, c.in, &rng)
                : MakeOp(c.m, c.k, c.pad, c.batch, false, c.in, &rng);
    Op b = c.tb ? MakeOp(c.n, c.k, c.pad, c.batch, c.broadcast_b, c.in, &rng)
                : MakeOp(c.k, c.n, c.pad, c.batch, c.broadcast_b, c.in, &rng);
    Op d = MakeOp(c.m, c.n, c.pad, c.batch, false, c.out, &rng);
    const int64_t d_elems =
        (c.batch - 1) * d.stride + (c.m - 1) * d.ld + c.n;
    std::vector<float> c0(d_elems);
    for (int64_t i = 0; i < d_elems; ++i) c0[i] = Load(d.alloc.ptr, i, c.out);

    GemmParams p;
    p.m = c.m;
    p.n = c.n;
    p.k = c.k;
    p.batch_count = c.batch;
    p.alpha = c.alpha;
    p.beta = c.beta;
    p.a = Bind(a, c.in, c.ta);
    p.b = Bind(b, c.in, c.tb);
    p.c = Bind(d, c.out, false);
    ASSERT_TRUE(SteelGemmSupports(p));
    absl::Status s = RunSteelGemm(device_, stream_, p, c.tile);
    ASSERT_TRUE(s.ok()) << s;
    ASSERT_TRUE(stream_->Synchronize().ok());

    int bad = 0;
    double max_err = 0;
    for (int64_t bt = 0; bt < c.batch; ++bt) {
      for (int64_t i = 0; i < c.m; ++i) {
        for (int64_t j = 0; j < c.n; ++j) {
          double acc = 0, mag = 0;
          for (int64_t kk = 0; kk < c.k; ++kk) {
            const int64_t ai = bt * a.stride +
                               (c.ta ? kk * a.ld + i : i * a.ld + kk);
            const int64_t bi = bt * b.stride +
                               (c.tb ? j * b.ld + kk : kk * b.ld + j);
            const double x = Load(a.alloc.ptr, ai, c.in) *
                             static_cast<double>(Load(b.alloc.ptr, bi, c.in));
            acc += x;
            mag += std::fabs(x);
          }
          const int64_t di = bt * d.stride + i * d.ld + j;
          double want = c.alpha * acc;
          mag = std::fabs(c.alpha) * mag;
          if (c.beta != 0.0) {
            want += c.beta * c0[di];
            mag += std::fabs(c.beta * c0[di]);
          }
          const double got = Load(d.alloc.ptr, di, c.out);
          const double err = std::fabs(got - want);
          max_err = std::max(max_err, err);
          const double tol = Eps(c.out) * std::fabs(want) +
                             (1e-5 + 1e-6 * c.k) * mag + 1e-6;
          if (!(err <= tol) && bad++ < 5) {
            ADD_FAILURE() << "batch " << bt << " (" << i << "," << j
                          << "): got " << got << " want " << want;
          }
        }
      }
      // Padding between rows must be untouched.
      for (int64_t i = 0; i + 1 < c.m; ++i) {
        for (int64_t j = c.n; j < d.ld; ++j) {
          const int64_t di = bt * d.stride + i * d.ld + j;
          if (Load(d.alloc.ptr, di, c.out) != c0[di] && bad++ < 5) {
            ADD_FAILURE() << "padding clobbered at " << di;
          }
        }
      }
    }
    EXPECT_EQ(bad, 0) << "max abs err " << max_err;
    for (Op* o : {&a, &b, &d}) ASSERT_TRUE(device_->Deallocate(o->alloc.ptr).ok());
  }

  static rt::Device* device_;
  static rt::Stream* stream_;
};
rt::Device* SteelGemmTest::device_ = nullptr;
rt::Stream* SteelGemmTest::stream_ = nullptr;

constexpr MpsDType kBF16 = MpsDType::kBF16;
constexpr MpsDType kF16 = MpsDType::kF16;
constexpr MpsDType kF32 = MpsDType::kF32;

TEST_F(SteelGemmTest, AlignedAllTransposes) {
  for (bool ta : {false, true})
    for (bool tb : {false, true}) {
      Case c{128, 128, 64};
      c.ta = ta;
      c.tb = tb;
      Run(c);
    }
}

TEST_F(SteelGemmTest, UnalignedShapes) {
  const int64_t shapes[][3] = {{1, 1, 1},    {3, 5, 7},     {65, 63, 17},
                               {100, 37, 50}, {33, 129, 8}, {130, 70, 300},
                               {7, 200, 1},  {64, 64, 15},  {257, 31, 33}};
  for (const auto& s : shapes)
    for (bool ta : {false, true})
      for (bool tb : {false, true}) {
        Case c{s[0], s[1], s[2]};
        c.ta = ta;
        c.tb = tb;
        Run(c);
      }
}

TEST_F(SteelGemmTest, DtypesAndOutputs) {
  const MpsDType combos[][2] = {
      {kBF16, kBF16}, {kBF16, kF32}, {kF16, kF16}, {kF16, kF32}, {kF32, kF32}};
  for (const auto& t : combos)
    for (int64_t m : {64, 77}) {
      Case c{m, 96, 40};
      c.in = t[0];
      c.out = t[1];
      c.tb = true;
      Run(c);
    }
}

TEST_F(SteelGemmTest, AlphaBeta) {
  for (MpsDType out : {kBF16, kF32})
    for (auto ab : {std::pair{2.0, 0.0}, {1.0, 1.0}, {0.5, -1.5}}) {
      for (int64_t m : {64, 45}) {
        Case c{m, 64, 32};
        c.out = out;
        c.alpha = ab.first;
        c.beta = ab.second;
        Run(c);
      }
    }
}

TEST_F(SteelGemmTest, ZeroKGivesBetaC) {
  Case c{20, 30, 0};
  c.beta = 0.5;
  Run(c);
  Case z{20, 30, 0};
  Run(z);
}

TEST_F(SteelGemmTest, BatchedStridedAndBroadcast) {
  Case c{50, 70, 30, 5};
  c.pad = 3;
  Run(c);
  c.broadcast_b = true;
  c.tb = true;
  Run(c);
  Case big{128, 128, 64, 16};
  big.beta = 1.0;
  Run(big);
}

TEST_F(SteelGemmTest, LeadingDimPadding) {
  for (bool ta : {false, true})
    for (bool tb : {false, true}) {
      Case c{64, 64, 48};
      c.pad = 5;
      c.ta = ta;
      c.tb = tb;
      Run(c);
    }
}

TEST_F(SteelGemmTest, AllTileConfigs) {
  static const SteelTile tiles[] = {{64, 64, 16, 1, 2}, {64, 64, 16, 2, 2},
                                    {64, 32, 32, 2, 2}, {32, 64, 16, 1, 2},
                                    {32, 32, 16, 2, 2}};
  for (const SteelTile& t : tiles)
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (int64_t m : {128, 97}) {
          Case c{m, 160, 72};
          c.ta = ta;
          c.tb = tb;
          c.tile = &t;
          c.beta = m == 97 ? 1.0 : 0.0;
          Run(c);
        }
}

TEST_F(SteelGemmTest, LargeK) {
  Case c{64, 64, 4096};
  c.out = kF32;
  Run(c);
}

TEST_F(SteelGemmTest, Supports) {
  GemmParams p;
  p.m = p.n = p.k = 16;
  p.a.ld = p.b.ld = p.c.ld = 16;
  p.a.dtype = p.b.dtype = p.c.dtype = kBF16;
  EXPECT_TRUE(SteelGemmSupports(p));
  p.a.dtype = kF16;
  EXPECT_FALSE(SteelGemmSupports(p));
  p.a.dtype = kBF16;
  p.a.ld = int64_t{1} << 23;  // > INT32_MAX / 256
  std::string why;
  EXPECT_FALSE(SteelGemmSupports(p, &why));
  EXPECT_EQ(why, "leading dimension too large");
}


// Timing, not correctness: min wall time over reps per tile config.
//   bazel run :steel_gemm_test -- --gtest_filter='*Benchmark*' \
//     --gtest_also_run_disabled_tests
TEST_F(SteelGemmTest, DISABLED_Benchmark) {
  struct Shape {
    int64_t m, n, k, batch;
    MpsDType in;
    bool tb;
  };
  const Shape shapes[] = {{2048, 2048, 2048, 1, kBF16, false},
                          {2048, 2048, 2048, 1, kBF16, true},
                          {2048, 2048, 2048, 1, kF16, false},
                          {512, 512, 512, 16, kBF16, false},
                          {4096, 4096, 4096, 1, kBF16, false},
                          {1024, 1024, 1024, 1, kBF16, false}};
  static const SteelTile tiles[] = {{64, 64, 16, 1, 2}, {64, 64, 16, 2, 2},
                                    {64, 32, 32, 2, 2}, {32, 64, 16, 1, 2},
                                    {32, 32, 16, 2, 2}, {64, 64, 32, 2, 2}};
  std::mt19937 rng(1);
  for (const Shape& sh : shapes) {
    Op a = MakeOp(sh.m, sh.k, 0, sh.batch, false, sh.in, &rng);
    Op b = sh.tb ? MakeOp(sh.n, sh.k, 0, sh.batch, false, sh.in, &rng)
                 : MakeOp(sh.k, sh.n, 0, sh.batch, false, sh.in, &rng);
    Op d = MakeOp(sh.m, sh.n, 0, sh.batch, false, sh.in, &rng);
    GemmParams p;
    p.m = sh.m;
    p.n = sh.n;
    p.k = sh.k;
    p.batch_count = sh.batch;
    p.a = Bind(a, sh.in, false);
    p.b = Bind(b, sh.in, sh.tb);
    p.c = Bind(d, sh.in, false);
    auto time = [&](auto&& launch) -> double {
      double best = 1e30;
      for (int r = 0; r < 12; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        absl::Status st = launch();
        if (st.ok()) st = stream_->Synchronize();
        if (!st.ok()) {
          ADD_FAILURE() << st;
          return -1.0;
        }
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (r > 1) best = std::min(best, ms);
      }
      return best;
    };
    const double flop = 2.0 * sh.m * sh.n * sh.k * sh.batch;
    auto report = [&](const std::string& what, double ms) {
      std::printf("%-10s %5lldx%5lldx%5lld b%-3lld %s%s  %8.3f ms  %6.2f TFLOP/s\n",
                  MpsDTypeName(sh.in), (long long)sh.m, (long long)sh.n,
                  (long long)sh.k, (long long)sh.batch, sh.tb ? "NT " : "NN ",
                  what.c_str(), ms, flop / ms / 1e9);
    };
    report("steel auto ", time([&] {
             return RunSteelGemm(device_, stream_, p);
           }));
    for (const SteelTile& t : tiles) {
      report(absl::StrCat("steel ", t.bm, "x", t.bn, "x", t.bk, "w", t.wm,
                          t.wn),
             time([&] { return RunSteelGemm(device_, stream_, p, &t); }));
    }
    for (Op* o : {&a, &b, &d}) ASSERT_TRUE(device_->Deallocate(o->alloc.ptr).ok());
  }
}

}  // namespace
}  // namespace blas
}  // namespace metal_pjrt
