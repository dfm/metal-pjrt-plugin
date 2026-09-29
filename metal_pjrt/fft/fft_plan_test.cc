// PlanFft, ElemsPerThread and LaunchGeometry against MLX's rules
// (mlx/backend/metal/fft.cpp, MLX 0.32.2) and the kernels' fixed-size
// arrays; the Rader and Bluestein constants by running those algorithms in
// double with them and comparing with a plain DFT; the workspace size. A
// host test: no Metal.
#include "metal_pjrt/fft/fft_plan.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"

namespace metal_pjrt {
namespace fft {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using cdouble = std::complex<double>;
using Steps = std::array<int, kNumRadices>;

// Steps from {radix: count}.
Steps S(std::initializer_list<std::pair<int, int>> counts) {
  Steps s = {};
  for (auto [radix, count] : counts) {
    for (int i = 0; i < kNumRadices; ++i) {
      if (kRadices[i] == radix) s[i] = count;
    }
  }
  return s;
}

FftPlan Plan(int64_t n) {
  absl::StatusOr<FftPlan> p = PlanFft(n);
  EXPECT_THAT(p, IsOk()) << n;
  return p.ok() ? *p : FftPlan{};
}

int64_t Product(const Steps& s) {
  int64_t p = 1;
  for (int i = 0; i < kNumRadices; ++i) {
    for (int j = 0; j < s[i]; ++j) p *= kRadices[i];
  }
  return p;
}

// sum_j z[j] exp(sign 2 pi i j k / n).
std::vector<cdouble> Dft(const std::vector<cdouble>& z, int sign = -1) {
  const int64_t n = z.size();
  std::vector<cdouble> out(n);
  for (int64_t k = 0; k < n; ++k) {
    cdouble sum = 0.0;
    for (int64_t j = 0; j < n; ++j) {
      sum += z[j] * internal::Twiddle(-sign * j * k, n);
    }
    out[k] = sum;
  }
  return out;
}

std::vector<cdouble> Random(int64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<cdouble> z(n);
  for (cdouble& v : z) v = {u(rng), u(rng)};
  return z;
}

double MaxRelError(const std::vector<cdouble>& y,
                   const std::vector<cdouble>& ref) {
  double err = 0, norm = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    err = std::max(err, std::abs(y[i] - ref[i]));
    norm = std::max(norm, std::abs(ref[i]));
  }
  return err / norm;
}

std::vector<cdouble> ReadComplex(const FftConstants& c, uint64_t offset,
                                 int64_t count) {
  std::vector<cdouble> v(count);
  for (int64_t i = 0; i < count; ++i) {
    float f[2];
    std::memcpy(f, c.bytes.data() + offset + 8 * i, 8);
    v[i] = {f[0], f[1]};
  }
  return v;
}

std::vector<int> ReadShort(const FftConstants& c, uint64_t offset,
                           int64_t count) {
  std::vector<int> v(count);
  for (int64_t i = 0; i < count; ++i) {
    int16_t s;
    std::memcpy(&s, c.bytes.data() + offset + 2 * i, 2);
    v[i] = s;
  }
  return v;
}

TEST(FftPlanTest, MlxPlans) {
  // Stockham: powers of two below 512 use radices 4 and 2 only.
  EXPECT_EQ(Plan(1).stockham, Steps{});
  EXPECT_EQ(Plan(8).stockham, S({{4, 1}, {2, 1}}));
  EXPECT_EQ(Plan(256).stockham, S({{4, 4}}));
  EXPECT_EQ(Plan(512).stockham, S({{8, 3}}));
  EXPECT_EQ(Plan(4096).stockham, S({{8, 4}}));
  EXPECT_EQ(Plan(60).stockham, S({{6, 1}, {5, 1}, {2, 1}}));
  EXPECT_EQ(Plan(3159).stockham, S({{13, 1}, {3, 5}}));
  // Rader: one prime factor p > 13 with p - 1 13-smooth.
  FftPlan p = Plan(17);
  EXPECT_EQ(p.rader_n, 17);
  EXPECT_EQ(p.rader, S({{4, 2}}));
  EXPECT_EQ(p.stockham, Steps{});
  p = Plan(1982);  // 2 * 991, 990 = 11 * 6 * 5 * 3
  EXPECT_EQ(p.rader_n, 991);
  EXPECT_EQ(p.rader, S({{11, 1}, {6, 1}, {5, 1}, {3, 1}}));
  EXPECT_EQ(p.stockham, S({{2, 1}}));
  EXPECT_EQ(p.bluestein_n, -1);
  // Fused Bluestein: p - 1 not smooth (47: 46 = 2 * 23), two Rader
  // factors (289 = 17^2).
  p = Plan(47);
  EXPECT_EQ(p.bluestein_n, 128);
  EXPECT_FALSE(p.four_step);
  EXPECT_EQ(p.rader, Steps{});
  EXPECT_EQ(p.stockham, S({{4, 3}, {2, 1}}));
  p = Plan(289);
  EXPECT_EQ(p.bluestein_n, 1024);
  EXPECT_FALSE(p.four_step);
  EXPECT_EQ(p.rader_n, 17);  // left over from the first factor, as MLX
  // Multi-upload Bluestein: a prime above 2048, any non-power of two above
  // 4096.
  p = Plan(2053);
  EXPECT_TRUE(p.four_step);
  EXPECT_EQ(p.bluestein_n, 8192);
  p = Plan(4099);
  EXPECT_TRUE(p.four_step);
  EXPECT_EQ(p.bluestein_n, 16384);
  p = Plan(6000);  // smooth, but above 4096
  EXPECT_TRUE(p.four_step);
  EXPECT_EQ(p.bluestein_n, 16384);
  // Four-step powers of two.
  for (auto [n, n1, n2] : {std::tuple{8192, 128, 64}, {65536, 1024, 64},
                           {131072, 128, 1024}, {1 << 23, 4096, 2048},
                           {1 << 24, 4096, 4096}}) {
    p = Plan(n);
    EXPECT_TRUE(p.four_step) << n;
    EXPECT_EQ(p.bluestein_n, -1) << n;
    EXPECT_EQ(p.n1, n1) << n;
    EXPECT_EQ(p.n2, n2) << n;
  }
  // The largest non-power of two has bluestein_n 2^24.
  EXPECT_EQ(Plan((1 << 23) - 1).bluestein_n, 1 << 24);
  for (int64_t n : {int64_t{0}, int64_t{-1}, int64_t{1} << 25,
                    (int64_t{1} << 23) + 1, (int64_t{1} << 24) + 1}) {
    EXPECT_THAT(PlanFft(n), StatusIs(absl::StatusCode::kUnimplemented)) << n;
  }
}

TEST(FftPlanTest, ElemsPerThread) {
  EXPECT_EQ(ElemsPerThread(Plan(4096)), 8);
  EXPECT_EQ(ElemsPerThread(Plan(256)), 4);
  EXPECT_EQ(ElemsPerThread(Plan(28)), 7);  // fft.h's example: 4 x 7
  EXPECT_EQ(ElemsPerThread(Plan(60)), 5);  // second smallest of {2, 5, 6}
  EXPECT_EQ(ElemsPerThread(Plan(1001)), 7);  // 7, 11, 13
  EXPECT_EQ(ElemsPerThread(Plan(143)), 11);  // 11, 13
  EXPECT_EQ(ElemsPerThread(Plan(22)), 6);    // (2 + 11) / 2
  // MLX's hand-tuned sizes.
  EXPECT_EQ(ElemsPerThread(Plan(3159)), 13);
  EXPECT_EQ(ElemsPerThread(Plan(3645)), 5);
  EXPECT_EQ(ElemsPerThread(Plan(3969)), 7);
  EXPECT_EQ(ElemsPerThread(Plan(1982)), 5);
}

// Every single-kernel plan and every four-step pass: the decomposition
// multiplies out, and the launch fits the kernels' arrays (13 inputs and
// 18 outputs per thread, MAX_RADIX / MAX_OUTPUT_SIZE), threadgroup memory
// (4096 complex64s) and 1024 threads per threadgroup.
TEST(FftPlanTest, EveryPlanFitsTheKernels) {
  auto check = [](const FftPlan& p, bool real, bool four_step) {
    SCOPED_TRACE(p.n);
    const FftLaunchGeometry g = LaunchGeometry(p, real, four_step, 1);
    EXPECT_LE(g.elems_per_thread, 13);
    for (int i = 0; i < kNumRadices; ++i) {
      if (p.stockham[i] > 0 || p.rader[i] > 0) {
        const int r = kRadices[i];
        EXPECT_LE((g.elems_per_thread + r - 1) / r * r, 18) << "radix " << r;
      }
    }
    EXPECT_GE(int64_t{g.elems_per_thread} * g.threads_per_fft, g.fft_size);
    EXPECT_LE(g.tg_mem_size, kMaxStockhamSize);
    EXPECT_GE(g.tg_mem_size, g.batch_per_group * g.fft_size);
    EXPECT_LE(g.batch_per_group * g.threads_per_fft, 1024);
    EXPECT_EQ(g.groups, 1);
  };
  for (int n = 2; n <= kMaxStockhamSize; ++n) {
    const FftPlan p = Plan(n);
    if (p.four_step) {
      EXPECT_GT(p.bluestein_n, kMaxStockhamSize) << n;
      continue;
    }
    if (p.bluestein_n > 0) {
      EXPECT_EQ(p.bluestein_n, [&] {
        int b = 1;
        while (b < 2 * n - 1) b *= 2;
        return b;
      }()) << n;
      EXPECT_EQ(Product(p.stockham), p.bluestein_n) << n;
    } else {
      EXPECT_EQ(Product(p.stockham) * p.rader_n, n) << n;
      if (p.rader_n > 1) EXPECT_EQ(Product(p.rader), p.rader_n - 1) << n;
    }
    for (bool real : {false, true}) check(p, real, false);
  }
  for (int64_t n = 2 * kMaxStockhamSize; n <= kMaxFftSize; n *= 2) {
    const FftPlan p = Plan(n);
    EXPECT_EQ(int64_t{p.n1} * p.n2, n);
    check(Plan(p.n1), false, true);
    check(Plan(p.n2), false, true);
  }
}

TEST(FftPlanTest, LaunchGeometry) {
  // n = 8: 32 transforms per threadgroup, 2 threads (4 elements) each.
  FftLaunchGeometry g = LaunchGeometry(Plan(8), false, false, 100);
  EXPECT_EQ(g.batch_per_group, 32);
  EXPECT_EQ(g.threads_per_fft, 2);
  EXPECT_EQ(g.tg_mem_size, 256);
  EXPECT_EQ(g.groups, 4);
  // Real transforms pack two rows into one.
  EXPECT_EQ(LaunchGeometry(Plan(8), true, false, 100).groups, 2);
  EXPECT_EQ(LaunchGeometry(Plan(8), true, false, 65).groups, 2);
  // Four-step passes batch at least 4 (coalescing), within 4096.
  g = LaunchGeometry(Plan(1024), false, true, 64);
  EXPECT_EQ(g.batch_per_group, 4);
  EXPECT_EQ(g.tg_mem_size, 4096);
  EXPECT_EQ(g.groups, 16);
  EXPECT_EQ(LaunchGeometry(Plan(4096), false, true, 1).batch_per_group, 1);
  // Non-power-of-two sizes round the threadgroup memory up.
  EXPECT_EQ(LaunchGeometry(Plan(300), false, false, 1).tg_mem_size, 512);
  // Bluestein runs at bluestein_n.
  g = LaunchGeometry(Plan(47), false, false, 1);
  EXPECT_EQ(g.fft_size, 128);
  EXPECT_EQ(g.batch_per_group, 2);
}

TEST(FftPlanTest, Pow2FftDouble) {
  for (int64_t n = 1; n <= 1024; n *= 2) {
    std::vector<cdouble> z = Random(n, 7 + n);
    const std::vector<cdouble> ref = Dft(z);
    internal::Pow2FftDouble(z);
    EXPECT_LT(MaxRelError(z, ref), 1e-13) << n;
  }
}

TEST(FftPlanTest, PrimitiveRoot) {
  EXPECT_EQ(internal::PrimitiveRoot(17), 3);
  EXPECT_EQ(internal::PrimitiveRoot(991), 6);
  EXPECT_EQ(internal::PrimitiveRoot(2039), 7);
  EXPECT_EQ(internal::PrimitiveRoot(97), 5);
}

// The Rader kernel's algorithm (rader_fft in kernels/fft.metal) in double
// with the float constants: x_perm = x[g_q]; y = fft(x_perm) * b_q;
// z = ifft(y) + x[0]; out[g_minus_q] = z; out[0] = sum(x).
TEST(FftPlanTest, RaderConstants) {
  for (int p : {17, 97, 257, 991, 2017, 2029}) {
    SCOPED_TRACE(p);
    const FftPlan plan = Plan(p);
    ASSERT_EQ(plan.rader_n, p);
    const FftConstants c = MakeFftConstants(plan);
    const int m = p - 1;
    const std::vector<cdouble> b_q = ReadComplex(c, c.b_q, m);
    const std::vector<int> g_q = ReadShort(c, c.g_q, m);
    const std::vector<int> g_minus_q = ReadShort(c, c.g_minus_q, m);
    EXPECT_EQ(c.b_q % 256, 0);
    EXPECT_EQ(c.g_q % 256, 0);
    EXPECT_EQ(c.g_minus_q % 256, 0);
    const std::vector<cdouble> x = Random(p, p);
    std::vector<cdouble> perm(m);
    for (int q = 0; q < m; ++q) perm[q] = x[g_q[q]];
    std::vector<cdouble> y = Dft(perm);
    for (int q = 0; q < m; ++q) y[q] *= b_q[q];
    std::vector<cdouble> z = Dft(y, +1);
    std::vector<cdouble> out(p);
    out[0] = 0;
    for (const cdouble& v : x) out[0] += v;
    for (int q = 0; q < m; ++q) out[g_minus_q[q]] = z[q] / double(m) + x[0];
    EXPECT_LT(MaxRelError(out, Dft(x)), 1e-6);
  }
}

// Bluestein's algorithm (bluestein_fft) in double with the float constants:
// out = w_k * ifft(fft(pad(w_k * x)) * w_q)[n - 1:2n - 1].
TEST(FftPlanTest, BluesteinConstants) {
  for (int n : {47, 289, 323, 1021, 2039, 2053, 4099, 10007}) {
    SCOPED_TRACE(n);
    const FftPlan plan = Plan(n);
    ASSERT_GT(plan.bluestein_n, 0);
    const int bn = plan.bluestein_n;
    const FftConstants c = MakeFftConstants(plan);
    EXPECT_EQ(c.w_k % 256, 0);
    const std::vector<cdouble> w_q = ReadComplex(c, c.w_q, bn);
    const std::vector<cdouble> w_k = ReadComplex(c, c.w_k, n);
    const std::vector<cdouble> x = Random(n, n);
    std::vector<cdouble> a(bn, 0.0);
    for (int k = 0; k < n; ++k) a[k] = x[k] * w_k[k];
    internal::Pow2FftDouble(a);
    for (int k = 0; k < bn; ++k) a[k] = std::conj(a[k] * w_q[k]);
    internal::Pow2FftDouble(a);  // conj(fft(conj(.))) = bn * ifft
    std::vector<cdouble> out(n);
    for (int k = 0; k < n; ++k) {
      out[k] = std::conj(a[n - 1 + k]) / double(bn) * w_k[k];
    }
    EXPECT_LT(MaxRelError(out, Dft(x)), 1e-6);
  }
}

// The chirp's angle pi k^2 / n is reduced exactly: w_k stays accurate for
// the largest n (the unreduced angle would be ~2^45 rad, off by ~1e-2).
TEST(FftPlanTest, BluesteinChirpLargeN) {
  const int64_t n = (int64_t{1} << 23) - 1;
  const FftConstants c = MakeFftConstants(Plan(n));
  for (int64_t k : {int64_t{1}, int64_t{12345}, n - 2, n - 1}) {
    const std::vector<cdouble> w = ReadComplex(c, c.w_k + 8 * k, 1);
    const double theta = -M_PI * static_cast<double>(k * k % (2 * n)) / n;
    EXPECT_NEAR(w[0].real(), std::cos(theta), 1e-7) << k;
    EXPECT_NEAR(w[0].imag(), std::sin(theta), 1e-7) << k;
  }
}

TEST(FftPlanTest, NoConstantsForStockhamAndFourStep) {
  EXPECT_TRUE(MakeFftConstants(Plan(1024)).bytes.empty());
  EXPECT_TRUE(MakeFftConstants(Plan(1 << 20)).bytes.empty());
  EXPECT_EQ(MakeFftConstants(Plan(1024)).size, 0);
  for (int64_t n : {17, 47, 2053}) {
    const FftConstants c = MakeFftConstants(Plan(n));
    EXPECT_EQ(c.size, c.bytes.size()) << n;
    EXPECT_GT(c.size, 0) << n;
  }
}

TEST(FftPlanTest, Workspace) {
  EXPECT_EQ(FftWorkspaceBytes(Plan(4096), 100), 0);
  EXPECT_EQ(FftWorkspaceBytes(Plan(2029), 100), 0);  // fused Bluestein
  EXPECT_EQ(FftWorkspaceBytes(Plan(8192), 3), 3 * 8192 * 8);
  EXPECT_EQ(FftWorkspaceBytes(Plan(8192), 0), 0);
  EXPECT_EQ(FftWorkspaceBytes(Plan(2053), 2), 2 * 2 * 8192 * 8);
  // Chunks of whole rows, at least one.
  EXPECT_EQ(ChunkRows(Plan(8192), 3, 2 * 8192 + 1), 2);
  EXPECT_EQ(FftWorkspaceBytes(Plan(8192), 3, 2 * 8192 + 1), 2 * 8192 * 8);
  EXPECT_EQ(ChunkRows(Plan(2053), 3, 100), 1);
  EXPECT_EQ(FftWorkspaceBytes(Plan(2053), 3, 100), 2 * 8192 * 8);
  EXPECT_EQ(ChunkRows(Plan(1 << 24), 5), 1);
  EXPECT_EQ(ChunkRows(Plan(64), 5), 5);
  EXPECT_EQ(ChunkRows(Plan(64), 1 << 20), (1 << 24) / 64);
}

}  // namespace
}  // namespace fft
}  // namespace metal_pjrt
