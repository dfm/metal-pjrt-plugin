// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Tests ChooseGemv / RunGemv (the wide gemv, gemv.h) through a
// metal_pjrt::rt::Stream against a CPU double reference (no XLA): both
// orientations, every vector count, tails, strides, batches, alpha/beta and
// the epilogue. Needs a Metal device.
#include "metal_pjrt/blas/gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace blas {
namespace {

constexpr MpsDType kBF16 = MpsDType::kBF16;
constexpr MpsDType kF16 = MpsDType::kF16;
constexpr MpsDType kF32 = MpsDType::kF32;

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
void Store(void* base, int64_t i, MpsDType t, float v) {
  switch (t) {
    case kF32: static_cast<float*>(base)[i] = v; break;
    case kF16:
      static_cast<_Float16*>(base)[i] = static_cast<_Float16>(v);
      break;
    case kBF16: static_cast<uint16_t*>(base)[i] = F32ToBf16(v); break;
  }
}
float Load(const void* base, int64_t i, MpsDType t) {
  switch (t) {
    case kF32: return static_cast<const float*>(base)[i];
    case kF16: return static_cast<float>(static_cast<const _Float16*>(base)[i]);
    case kBF16: return Bf16ToF32(static_cast<const uint16_t*>(base)[i]);
  }
  return 0;
}
double Eps(MpsDType t) {
  return t == kF32 ? 1e-6 : t == kF16 ? 1.0 / 1024 : 1.0 / 128;
}
double Act(int act, double x) {
  if (act == 1) return x > 0 ? x : 0;
  if (act == 2) {
    const double k = 0.7978845608028654;  // sqrt(2/pi)
    return 0.5 * x * (1 + std::tanh(k * (x + 0.044715 * x * x * x)));
  }
  if (act == 3) return x / (1 + std::exp(-x));
  return x;
}

struct Case {
  int64_t m, n, k, batch = 1;
  MpsDType in = kBF16, out = kBF16;
  double alpha = 1.0, beta = 0.0;
  int64_t pad = 0;           // extra elements per stored row (multiple of 4)
  int64_t d_pad = 0;         // extra elements per row of D (any)
  bool broadcast_b = false;  // B batch stride 0
  int act = 0;
  bool bias = false, aux = false;
  std::string Name() const {
    return absl::StrCat(MpsDTypeName(in), "->", MpsDTypeName(out), " ", m,
                        "x", n, "x", k, " b", batch, " a", alpha, " b", beta,
                        " pad", pad, "/", d_pad, broadcast_b ? " bcastB" : "",
                        " act", act, bias ? " bias" : "", aux ? " aux" : "");
  }
};

class GemvTest : public ::testing::Test {
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
    int64_t ld, stride, elems;
  };

  // rows x cols stored row-major with `pad`, `batches` apart (0 = one copy).
  Op MakeOp(int64_t rows, int64_t cols, int64_t pad, int64_t batches,
            bool broadcast, MpsDType t, std::mt19937* rng) {
    Op op{{}, cols + pad, broadcast ? 0 : rows * (cols + pad), 0};
    const int64_t nb = broadcast ? 1 : batches;
    op.elems = (nb - 1) * op.stride + (rows - 1) * op.ld + cols;
    auto a = device_->Allocate(op.elems * MpsDTypeSize(t));
    EXPECT_TRUE(a.ok()) << a.status();
    op.alloc = *a;
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int64_t i = 0; i < op.elems; ++i) Store(op.alloc.ptr, i, t, u(*rng));
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

  // D = A B^T (A m x k, B stored n x k), the form ChooseGemv takes.
  void Run(const Case& c) {
    if (device_->info().gpu_family < 9) {
      GTEST_SKIP() << "the gemv needs Apple9 (M3) or later, as in MLX";
    }
    SCOPED_TRACE(c.Name());
    std::mt19937 rng(1234);
    Op a = MakeOp(c.m, c.k, c.pad, c.batch, false, c.in, &rng);
    Op b = MakeOp(c.n, c.k, c.pad, c.batch, c.broadcast_b, c.in, &rng);
    Op d = MakeOp(c.m, c.n, c.d_pad, c.batch, false, c.out, &rng);
    Op bias = MakeOp(1, c.n, 0, 1, true, c.out, &rng);
    Op aux = MakeOp(c.m, c.n, c.d_pad, c.batch, false, c.out, &rng);
    std::vector<float> d0(d.elems), aux0(aux.elems);
    for (int64_t i = 0; i < d.elems; ++i) d0[i] = Load(d.alloc.ptr, i, c.out);
    for (int64_t i = 0; i < aux.elems; ++i) {
      aux0[i] = Load(aux.alloc.ptr, i, c.out);
    }

    GemmParams p;
    p.m = c.m;
    p.n = c.n;
    p.k = c.k;
    p.batch_count = c.batch;
    p.alpha = c.alpha;
    p.beta = c.beta;
    p.a = Bind(a, c.in, false);
    p.b = Bind(b, c.in, true);
    p.c = Bind(d, c.out, false);
    const std::optional<GemvPlan> plan =
        ChooseGemv(p, device_->info().gpu_family);
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->vectors_are_b, c.n < c.m);
    SteelEpilogue epi;
    epi.act = c.act;
    if (c.bias) epi.bias = bias.alloc.ptr;
    if (c.aux) epi.aux = aux.alloc.ptr;
    absl::Status s = RunGemv(device_, stream_, p, *plan, &epi);
    ASSERT_TRUE(s.ok()) << s;
    ASSERT_TRUE(stream_->Synchronize().ok());

    int bad = 0;
    auto expect = [&](bool ok, const std::string& what) {
      if (!ok && bad++ < 5) ADD_FAILURE() << what;
    };
    for (int64_t bt = 0; bt < c.batch; ++bt) {
      for (int64_t i = 0; i < c.m; ++i) {
        for (int64_t j = 0; j < c.n; ++j) {
          double acc = 0, mag = 0;
          for (int64_t kk = 0; kk < c.k; ++kk) {
            const double x =
                Load(a.alloc.ptr, bt * a.stride + i * a.ld + kk, c.in) *
                static_cast<double>(
                    Load(b.alloc.ptr, bt * b.stride + j * b.ld + kk, c.in));
            acc += x;
            mag += std::fabs(x);
          }
          const int64_t di = bt * d.stride + i * d.ld + j;
          double v = c.alpha * acc;
          mag = std::fabs(c.alpha) * mag;
          if (c.beta != 0.0) {
            v += c.beta * d0[di];
            mag += std::fabs(c.beta * d0[di]);
          }
          if (c.bias) v += Load(bias.alloc.ptr, j, c.out);
          const double tol = (1e-5 + 1e-6 * c.k) * mag + 1e-6;
          const double want = Act(c.act, v);
          const double got = Load(d.alloc.ptr, di, c.out);
          expect(std::fabs(got - want) <=
                     tol + Eps(c.out) * (std::fabs(want) + 1e-3),
                 absl::StrCat("D batch ", bt, " (", i, ",", j, "): got ",
                              got, " want ", want));
          if (c.aux) {
            const double got_aux = Load(aux.alloc.ptr, di, c.out);
            expect(std::fabs(got_aux - v) <=
                       tol + Eps(c.out) * (std::fabs(v) + 1e-3),
                   absl::StrCat("aux (", i, ",", j, "): got ", got_aux,
                                " want ", v));
          }
        }
        // Padding between rows must be untouched (D and aux).
        for (int64_t j = c.n; j < d.ld && i + 1 < c.m; ++j) {
          const int64_t di = bt * d.stride + i * d.ld + j;
          expect(Load(d.alloc.ptr, di, c.out) == d0[di],
                 absl::StrCat("D padding clobbered at ", di));
          expect(Load(aux.alloc.ptr, di, c.out) == aux0[di],
                 absl::StrCat("aux padding clobbered at ", di));
        }
      }
    }
    if (!c.aux) {
      for (int64_t i = 0; i < aux.elems; ++i) {
        expect(Load(aux.alloc.ptr, i, c.out) == aux0[i], "aux written");
      }
    }
    EXPECT_EQ(bad, 0);
    for (Op* o : {&a, &b, &d, &bias, &aux}) {
      ASSERT_TRUE(device_->Deallocate(o->alloc.ptr).ok());
    }
  }

  static rt::Device* device_;
  static rt::Stream* stream_;
};
rt::Device* GemvTest::device_ = nullptr;
rt::Stream* GemvTest::stream_ = nullptr;

// Every vector count, in both orientations (m small: x W^T; n small: the
// vectors are B's rows and D is written through its column stride), with
// tail rows (not a multiple of 4) and K with a remainder both below (32 K
// lanes) and above one unrolled block (KL * 8 * 4 elements).
TEST_F(GemvTest, VectorCountsAndOrientations) {
  for (int64_t vecs = 2; vecs <= kGemvMaxVectors; ++vecs) {
    for (int64_t k : {516, 1028}) {
      Run(Case{vecs, 301, k});
      Run(Case{301, vecs, k});
    }
  }
}

TEST_F(GemvTest, TypesAndOutputs) {
  const MpsDType combos[][2] = {
      {kBF16, kBF16}, {kBF16, kF32}, {kF16, kF16}, {kF16, kF32}};
  for (const auto& t : combos) {
    for (int64_t m : {3, 8}) {
      Case c{m, 96, 520};
      c.in = t[0];
      c.out = t[1];
      Run(c);
      std::swap(c.m, c.n);
      Run(c);
    }
  }
}

// Padded rows, D padded by an odd amount, batches with and without a
// broadcast matrix, alpha / beta (C read from D in place).
TEST_F(GemvTest, StridesBatchesAlphaBeta) {
  for (bool swap : {false, true}) {
    Case c{5, 70, 512};
    c.pad = 4;
    c.d_pad = 3;
    c.batch = 3;
    c.alpha = 0.5;
    c.beta = -1.5;
    if (swap) std::swap(c.m, c.n);
    Run(c);
    c.broadcast_b = true;  // B is the matrix (m small) or the vectors
    Run(c);
  }
}

TEST_F(GemvTest, Epilogues) {
  for (bool swap : {false, true}) {
    for (int act : {0, 1, 2, 3}) {
      Case c{6, 40, 532};
      c.act = act;
      c.bias = true;
      c.aux = act != 0;
      c.batch = 2;
      c.d_pad = 1;
      if (swap) std::swap(c.m, c.n);
      Run(c);
    }
  }
  Case c{4, 33, 516};
  c.in = c.out = kF16;
  c.bias = true;
  c.beta = 1.0;
  Run(c);
}

// Rows >= 65536: one grid column walks every pass (MLX's vocabulary-wide
// case), here with two passes of four and three vectors.
TEST_F(GemvTest, WideMatrix) {
  Case c{7, 65540, 512};
  Run(c);
}

TEST_F(GemvTest, Plans) {
  GemmParams p;
  p.m = 2;
  p.n = 6144;
  p.k = 1024;
  p.a.dtype = p.b.dtype = p.c.dtype = kBF16;
  p.a.ld = p.b.ld = 1024;
  p.c.ld = 6144;
  p.b.transpose = true;
  // MLX's gemv_wide_config: passes = ceil(vecs / 5), vecs_per_tg =
  // ceil(vecs / passes), 32 K lanes with one pass (or <= 64 rows), else 16.
  const int want[][3] = {{2, 2, 32}, {5, 5, 32}, {6, 3, 16}, {8, 4, 16}};
  for (const auto& w : want) {
    p.m = w[0];
    std::optional<GemvPlan> plan = ChooseGemv(p, 9);
    ASSERT_TRUE(plan.has_value()) << w[0];
    EXPECT_FALSE(plan->vectors_are_b);
    EXPECT_EQ(plan->vecs_per_tg, w[1]) << w[0];
    EXPECT_EQ(plan->k_lanes, w[2]) << w[0];
    EXPECT_EQ(plan->grid_x, (w[0] + 4) / 5) << w[0];
  }
  p.m = 8;
  p.n = p.c.ld = 151936;  // vocabulary-wide: one grid column
  ASSERT_TRUE(ChooseGemv(p, 9).has_value());
  EXPECT_EQ(ChooseGemv(p, 9)->grid_x, 1);
  p.n = p.c.ld = 6144;
  EXPECT_TRUE(ChooseGemv(p, 9).has_value());

  // Not taken: steel keeps these.
  auto refuse = [&](auto&& change, const char* what) {
    GemmParams q = p;
    change(q);
    EXPECT_FALSE(ChooseGemv(q, 9).has_value()) << what;
  };
  refuse([](GemmParams& q) { q.m = 1; }, "one vector");
  refuse([](GemmParams& q) { q.m = 9; }, "9 vectors");
  refuse([](GemmParams& q) { q.m = q.n = 64; }, "both sides large");
  refuse([](GemmParams& q) { q.k = 1022; }, "K % 4");
  refuse([](GemmParams& q) { q.a.ld = 1030; }, "ld % 4");
  refuse([](GemmParams& q) { q.b.offset = 4; }, "offset % 8");
  refuse([](GemmParams& q) { q.a.transpose = true; }, "A transposed");
  refuse([](GemmParams& q) { q.b.transpose = false; }, "B not transposed");
  refuse([](GemmParams& q) { q.a.dtype = q.b.dtype = q.c.dtype = kF32; },
         "f32");
  refuse([](GemmParams& q) { q.k = 0; }, "k = 0");
  // K below kGemvMinK: batched decode attention, 1024 x [8, 128] x
  // [128, 128]^T, runs 1.9x faster on steel's 16-row tile.
  refuse([](GemmParams& q) {
    q.n = q.c.ld = q.k = q.a.ld = q.b.ld = 128;
    q.batch_count = 1024;
    q.a.batch_stride = q.c.batch_stride = 8 * 128;
    q.b.batch_stride = 128 * 128;
  }, "K = 128");
  refuse([](GemmParams& q) { q.k = q.a.ld = q.b.ld = 508; }, "K = 508");
  {
    GemmParams q = p;
    q.k = q.a.ld = q.b.ld = 512;
    EXPECT_TRUE(ChooseGemv(q, 9).has_value()) << "K = 512";
  }
  refuse([](GemmParams& q) { q.c.ld = 6000; }, "ldc < n");
  refuse([](GemmParams& q) {
    q.batch_count = 2;
    q.a.batch_stride = 2 * 1024 + 2;
  }, "batch stride % 4");
  EXPECT_FALSE(ChooseGemv(p, 8).has_value()) << "pre-M3";
}

}  // namespace
}  // namespace blas
}  // namespace metal_pjrt
